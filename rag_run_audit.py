"""
RAG + LLM 事实验证 - 训练集审计：前N条训练集走完整流水线，检验检索/标签正确率，统计各模型调用次数。
用法:
  python rag_run_audit.py                                # 默认: 从 submission_test.csv 已有进度追加续跑
  python rag_run_audit.py --next 0 --workers 4           # 追加至剩余全部
  python rag_run_audit.py --fresh --workers 4            # 忽略进度, 全量重跑
  python rag_run_audit.py --eval-train 100 --workers 4   # 审计前100条train, 总共1000条
"""
import csv, sys, re, time, os, argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from rag_fact_verify import load_corpus, make_idf, retrieve, tokenize, predict_label
from rag_llm_api import OpinionAnalyzer

csv.field_size_limit(sys.maxsize)

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 数据文件相对本脚本定位, 任意目录启动均可

# ====================== 可调参数 (在此修改) ======================
# 追加模式: 从 submission_test.csv 已有行数往后预测 N 条; 0=预测剩余全部; 不足 N 则预测剩余所有
PREDICT_NEXT_N = 100
APPEND_TO_CSV = 'submission_test.csv'   # 追加结果的目标CSV
TEST_CSV = 'test.csv'
# =================================================================

PROMPT_TMPL = """You are a fact verification assistant. Based ONLY on the given passage from a scientific paper, judge whether the statement is true or false.

Rules:
- 1 = the statement is consistent with the passage (correct paraphrase, faithful summary)
- 0 = the statement contradicts, exaggerates, over-generalizes, or alters numbers/quantifiers/negation compared with the passage

Output ONLY a single digit (1 or 0), nothing else.

Passage:
{sec}

Statement: {stmt}"""


_HARD_POOL = ThreadPoolExecutor(max_workers=64)
LLM_HARD_TIMEOUT = 150  # 单次调用硬超时(秒), 防止个别请求挂起


def preprocess_sec(analyzer, stmt, sec_text):
    """辅助模型(freellm-auto, 即原 llama-3.3 位置): 提取与陈述相关的关键句, 供主力模型聚焦"""
    prompt = ("From the passage below, extract ONLY the sentences most relevant for verifying the statement. "
              "Output the sentences verbatim, nothing else.\n\n"
              f"Passage:\n{sec_text}\n\nStatement: {stmt}")
    r = analyzer.analyze(prompt, model_id='auto')
    if r.get('success') and r.get('content'):
        return r['content'].strip()[:1500]
    return ''


def llm_label(analyzer, stmt, sec_text, max_sec_chars, preprocess=False, model_id=None):
    """返回 (label|None, model, prompt_chars)。失败自动重试一次, 再回退 pattern"""
    if not sec_text:
        return None, 'no_sec', 0
    key = preprocess_sec(analyzer, stmt, sec_text[:max_sec_chars]) if preprocess else ''
    prompt = PROMPT_TMPL.format(sec=sec_text[:max_sec_chars], stmt=stmt)
    if key:
        prompt += "\n\nAssistant-highlighted key sentences:\n" + key
    for _ in range(2):  # 重试一次, 吸收超时/限流/content=None 等瞬态故障
        try:
            r = _HARD_POOL.submit(analyzer.analyze, prompt, model_id=model_id).result(timeout=LLM_HARD_TIMEOUT)
        except Exception:
            r = None
        if r and r.get('success'):
            m = re.search(r'[01]', (r.get('content') or '').strip())
            if m:
                return int(m.group()), r['model'], len(prompt)
    return None, 'fail', len(prompt)


def process_item(item, docs, idf, analyzer, max_sec_chars, preprocess=False, model_id=None):
    stmt = item.get('raw_input') or item.get('ans', '')
    t1 = time.time()
    qset = set(tokenize(stmt))
    pd, ps, st = retrieve(qset, docs, idf)
    t_retr = time.time() - t1
    if pd is None:  # 检索失败兜底
        pd = next(iter(docs))
        ps = docs[pd]['sec_toks'][0][0] if docs[pd]['sec_toks'] else ''
        st = docs[pd]['sec_text'].get(ps, '')
    t2 = time.time()
    label, model, pchars = llm_label(analyzer, stmt, st, max_sec_chars, preprocess, model_id)
    t_llm = time.time() - t2
    src = 'llm'
    if label is None:
        label = predict_label(stmt, st)
        src = 'pattern'
    return {'stmt': stmt, 'review_id': pd, 'section_name': ps, 'label': label,
            'gold_review_id': item.get('review_id', ''), 'gold_section': item.get('section_name', ''),
            'gold_label': item.get('label', ''), 'ID': item.get('ID', ''),
            't_retr': t_retr, 't_llm': t_llm, 'src': src, 'model': model, 'pchars': pchars}


RAW_FIELDS = ['ID', 'gold_review_id', 'gold_section', 'gold_label', 'review_id',
              'section_name', 'label', 'model', 'src', 't_retr', 't_llm', 'pchars', 'stmt']


def save_rows(results, path, fields):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, '') for k in fields} for r in results)


def _cleanup_ckpt(path='ckpt_tmp.csv'):
    if os.path.exists(path):
        os.remove(path)


def run(items, docs, idf, analyzer, workers, max_sec_chars, ckpt_path=None, preprocess=False, model_id=None):
    print(f"待处理 {len(items)} 条, workers={workers}\n")
    t1 = time.time()
    results = []

    def _emit(i, r):
        results.append(r)
        print(f"[{i+1}/{len(items)}] retr={r['t_retr']:.2f}s llm={r['t_llm']:.1f}s "
              f"src={r['src']} model={r['model']} label={r['label']}")
        if ckpt_path and (i + 1) % 25 == 0:
            save_rows(results, ckpt_path, RAW_FIELDS)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_item, it, docs, idf, analyzer, max_sec_chars, preprocess, model_id)
                    for it in items]
            for i, f in enumerate(futs):
                _emit(i, f.result())
    else:
        for i, it in enumerate(items):
            _emit(i, process_item(it, docs, idf, analyzer, max_sec_chars, preprocess, model_id))
    return results, time.time() - t1


def model_stats(results, wall, workers):
    n = len(results)
    avg_retr = sum(r['t_retr'] for r in results) / n
    avg_llm = sum(r['t_llm'] for r in results) / n
    avg_pchars = sum(r['pchars'] for r in results) / n
    mc = Counter(r['model'] for r in results if r['src'] == 'llm')
    n_fallback = sum(1 for r in results if r['src'] == 'pattern')
    print(f"\n===== 统计 (n={n}, 墙钟 {wall:.0f}s) =====")
    print(f"单条耗时: 检索 {avg_retr:.2f}s + LLM {avg_llm:.1f}s; 回退pattern {n_fallback} 条")
    print("各模型调用次数:")
    for m, c in mc.most_common():
        print(f"  {m}: {c}")
    print(f"单条token: 输入≈{avg_pchars/4:.0f} + 输出≈10")


def eval_train(results):
    n = len(results)
    retr_ok = sum(1 for r in results
                  if r['review_id'] == r['gold_review_id'] and r['section_name'] == r['gold_section'])
    label_all = sum(1 for r in results if str(r['label']) == str(r['gold_label']))
    label_on_retr = sum(1 for r in results if str(r['label']) == str(r['gold_label'])
                        and r['review_id'] == r['gold_review_id'] and r['section_name'] == r['gold_section'])
    e2e = sum(1 for r in results if str(r['label']) == str(r['gold_label'])
              and r['review_id'] == r['gold_review_id'] and r['section_name'] == r['gold_section'])
    pct_retr = f"{label_on_retr/retr_ok:.1%}" if retr_ok else 'N/A'
    print(f"""
===== 训练集审计 (n={n}) =====
检索正确 (review_id+section): {retr_ok}/{n} = {retr_ok/n:.1%}
标签正确 (label):              {label_all}/{n} = {label_all/n:.1%}
标签正确率 （检索正确前提）:                {label_on_retr}/{retr_ok} = {pct_retr}
端到端正确率 (Kaggle 评分口径):     {e2e}/{n} = {e2e/n:.1%}""")

    wrong = [r for r in results if str(r['label']) != str(r['gold_label'])
             or r['review_id'] != r['gold_review_id']][:5]
    for r in wrong:
        retr = 'OK' if (r['review_id'] == r['gold_review_id'] and r['section_name'] == r['gold_section']) else 'MISS'
        print(f"  [错例 retr={retr}] gold={r['gold_label']} pred={r['label']} | {r['stmt'][:80]}")
    return e2e / n


def _count_done(append_csv):
    if not os.path.exists(append_csv):
        return 0
    with open(append_csv, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    return max(0, len(rows) - 1) if rows else 0


def run_append(docs, idf, analyzer, workers, max_sec_chars, n_next, append_csv,
               preprocess=False, model_id=None):
    start = _count_done(append_csv)
    test = list(csv.DictReader(open(TEST_CSV, encoding='utf-8')))
    remaining = test[start:]
    batch = remaining[:n_next] if n_next > 0 else remaining
    if not batch:
        print(f"{append_csv} 已含 {start}/{len(test)} 条, 无可预测数据。")
        return
    print(f"已完成 {start}/{len(test)}, 本次预测 {len(batch)} 条 (行 {start+1}~{start+len(batch)})\n")
    results, wall = run(batch, docs, idf, analyzer, workers, max_sec_chars,
                       ckpt_path='ckpt_tmp.csv', preprocess=preprocess, model_id=model_id)
    file_exists = os.path.exists(append_csv) and os.path.getsize(append_csv) > 0
    with open(append_csv, 'a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ID', 'review_id', 'section_name', 'label'])
        if not file_exists:
            w.writeheader()
        w.writerows({k: r[k] for k in ['ID', 'review_id', 'section_name', 'label']} for r in results)
    print(f"\n已追加 {len(results)} 条到 {append_csv} (累计 {start+len(results)}/{len(test)})")
    _cleanup_ckpt()
    model_stats(results, wall, workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='test前N条, 0=全量')
    ap.add_argument('--eval-train', type=int, default=0, help='train审计条数, 0=跳过')
    ap.add_argument('--train-start', type=int, default=0, help='train起始行(0-based)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--max-sec-chars', type=int, default=3500)
    ap.add_argument('--no-preprocess', action='store_true', help='关闭Llama辅助关键句提取')
    ap.add_argument('--model-id', default=None, help='指定判定模型ID(默认走流水线)')
    ap.add_argument('--out', default=None, help='test输出文件路径(默认按limit自动命名)')
    ap.add_argument('--fresh', action='store_true',
                    help='全新全量预测: 忽略 submission_test.csv 已有进度, 从 test 第1条重跑(默认追加续跑)')
    ap.add_argument('--next', type=int, default=PREDICT_NEXT_N,
                    help=f'追加模式预测条数(0=剩余全部; 默认用顶部 PREDICT_NEXT_N={PREDICT_NEXT_N})')
    ap.add_argument('--append-csv', default=APPEND_TO_CSV,
                    help=f'追加目标CSV(默认用顶部 APPEND_TO_CSV={APPEND_TO_CSV})')
    args = ap.parse_args()

    t0 = time.time()
    print("正在加载语料...")
    docs, df = load_corpus('original_data.csv')
    idf = make_idf(df, len(docs))
    print(f"已加载 {len(docs)} 篇文献, 用时 {time.time()-t0:.0f}s\n")
    analyzer = OpinionAnalyzer(use_freeflow=False)

    if args.eval_train > 0:
        all_rows = list(csv.DictReader(open('train.csv', encoding='utf-8')))
        train = all_rows[args.train_start:args.train_start + args.eval_train]
        results, wall = run(train, docs, idf, analyzer, args.workers, args.max_sec_chars,
                            ckpt_path='ckpt_tmp.csv', preprocess=not args.no_preprocess,
                            model_id=args.model_id)
        out = f'train_eval_{args.train_start+1}_{args.train_start+len(train)}.csv'
        save_rows(results, out, RAW_FIELDS)  # 先存明细再统计, 防统计崩溃丢结果
        print(f"\n明细已保存 {out}")
        _cleanup_ckpt()
        model_stats(results, wall, args.workers)
        eval_train(results)
        return

    if not args.fresh:  # 默认: 追加续跑, 防止重复调用浪费额度
        run_append(docs, idf, analyzer, args.workers, args.max_sec_chars,
                   args.next, args.append_csv,
                   preprocess=not args.no_preprocess, model_id=args.model_id)
        return

    test = list(csv.DictReader(open('test.csv', encoding='utf-8')))
    if args.limit > 0:
        test = test[:args.limit]
    results, wall = run(test, docs, idf, analyzer, args.workers, args.max_sec_chars,
                        ckpt_path='ckpt_tmp.csv', preprocess=not args.no_preprocess,
                        model_id=args.model_id)
    out = args.out or ('submission.csv' if args.limit == 0 else f'submission_test{len(test)}.csv')
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ID', 'review_id', 'section_name', 'label'])
        w.writeheader()
        w.writerows({k: r[k] for k in ['ID', 'review_id', 'section_name', 'label']} for r in results)
    print(f"\n已保存 {out}")
    _cleanup_ckpt()
    model_stats(results, wall, args.workers)


if __name__ == '__main__':
    main()
