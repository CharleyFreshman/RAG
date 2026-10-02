"""
基于文献的事实验证系统 - 模式匹配·改进版
核心方法：仍为关键词/IDF模式匹配。从原始文献(original_data.csv)检索 review_id 与 section_name，
        替代原版本"用训练陈述匹配测试陈述"的错误检索目标，未引入大模型。
"""
import csv, sys, re, math, ast, time
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as STOPWORDS  # 内置停用词表
except ImportError:  # sklearn 不可用时退化为仅长度过滤
    STOPWORDS = set()

TOKEN_RE = re.compile(r"[a-z]+")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
NEG_RE = re.compile(r"\b(not|no|never|neither|nor|cannot|none|without|barely|hardly|scarcely|lack|lacks|lacking|fail|fails|failed|rather\s+than|instead\s+of|unlike)\b")
REVERSAL_RE = re.compile(r"\b(rather\s+than|instead\s+of|not\s+but|not\s+only)\b")
QUANT_RE = re.compile(r"\b(all|every|each|only|exactly|solely|exclusively|always|never|none|both|most|majority|minority|no|without)\b")


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in STOPWORDS]


def parse_full_text(ft):
    try:
        d = ast.literal_eval(ft)
    except Exception:
        return []
    secs = d.get('section_name', []) or []
    paras = d.get('paragraphs', []) or []
    out = []
    for i, sname in enumerate(secs):
        p = paras[i] if i < len(paras) else []
        text = " ".join(p) if isinstance(p, list) else str(p)
        out.append((sname or '', text or ''))
    return out


def load_corpus(filepath):
    docs = {}
    df = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            secs = parse_full_text(r['full_text'])
            full = r['title'] + " " + r['abstract'] + " " + " ".join(t for _, t in secs)
            doc_toks = set(tokenize(full))
            ta_toks = set(tokenize(r['title'] + " " + r['abstract']))
            sec_toks = [(s, set(tokenize(t))) for s, t in secs if s]
            sec_text = {s: t for s, t in secs if s}
            for t in doc_toks:
                df[t] += 1
            docs[r['id']] = {
                'doc_toks': doc_toks, 'ta_toks': ta_toks,
                'sec_toks': sec_toks, 'sec_text': sec_text,
            }
    return docs, df


def make_idf(df, N):
    cache = {}
    def idf(t):
        if t not in cache:
            cache[t] = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
        return cache[t]
    return idf


def retrieve(qset, docs, idf):
    """返回 (review_id, section_name, section_text)。V5: 0.5*doc + 1.0*maxsec + 0.5*ta"""
    best_doc = None
    best_doc_score = 0.0
    doc_maxsec = {}
    for did, d in docs.items():
        dov = sum(idf(t) for t in qset if t in d['doc_toks'])
        if dov <= 0:
            continue
        ta = sum(idf(t) for t in qset if t in d['ta_toks'])
        maxs = 0.0
        for _, stoks in d['sec_toks']:
            ov = sum(idf(t) for t in qset if t in stoks)
            if ov > maxs:
                maxs = ov
        score = 0.5 * dov + 1.0 * maxs + 0.5 * ta
        doc_maxsec[did] = maxs
        if score > best_doc_score:
            best_doc_score = score
            best_doc = did
    if best_doc is None:
        return None, None, ''
    # section within predicted doc
    best_sec = None
    best_sec_score = -1.0
    for sname, stoks in docs[best_doc]['sec_toks']:
        ov = sum(idf(t) for t in qset if t in stoks)
        if ov > best_sec_score:
            best_sec_score = ov
            best_sec = sname
    sec_text = docs[best_doc]['sec_text'].get(best_sec or '', '')
    return best_doc, best_sec, sec_text


def predict_label(statement, sec_text):
    """基于扭曲信号的模式匹配：检测数字/量词/反转/否定等失真特征。无文本依据时强势判 0。
    调参依据 train 网格搜索（best_label_params.json）：label acc≈0.63。"""
    if not sec_text:
        return 0
    s_low = statement.lower()
    t_low = sec_text.lower()

    num_missing = bool(set(NUM_RE.findall(statement)) - set(NUM_RE.findall(sec_text)))
    neg_mismatch = bool(NEG_RE.search(s_low)) != bool(NEG_RE.search(t_low))
    reversal_mismatch = bool(REVERSAL_RE.search(s_low)) and not bool(REVERSAL_RE.search(t_low))
    quant_extra = bool(set(QUANT_RE.findall(s_low)) - set(QUANT_RE.findall(t_low)))

    score = 0
    if num_missing:
        score += -3
    if neg_mismatch:
        score += -2
    if reversal_mismatch:
        score += -3
    if quant_extra:
        score += -3
    return 1 if score >= -2 else 0


def evaluate_on_train(docs, idf):
    train = list(csv.DictReader(open('train.csv', encoding='utf-8')))
    both = 0
    label_correct = 0
    scored = 0
    pred_dist = Counter()
    gold_dist = Counter()
    for it in train:
        qset = set(tokenize(it['ans']))
        if not qset:
            continue
        pd, ps, st = retrieve(qset, docs, idf)
        scored += 1
        retr_ok = (pd == it['review_id'] and ps == it['section_name'])
        pl = predict_label(it['ans'], st)
        pred_dist[pl] += 1
        gold_dist[int(it['label'])] += 1
        if retr_ok and pl == int(it['label']):
            label_correct += 1
        if retr_ok:
            both += 1
    n = len(train)
    print(f"[train self-eval] retrieval both correct: {both}/{n} = {both/n:.3f}")
    print(f"[train self-eval] end-to-end (retr OK & label OK): {label_correct}/{n} = {label_correct/n:.3f}")
    print(f"[train self-eval] gold label dist: {dict(gold_dist)}")
    print(f"[train self-eval] pred label dist: {dict(pred_dist)}")
    if both:
        print(f"[train self-eval] label acc | retr OK: {label_correct}/{both} = {label_correct/both:.3f}")


def main(limit=0, out='submission.csv'):
    start = time.time()
    print("正在加载原始文献数据...")
    docs, df = load_corpus('original_data.csv')
    idf = make_idf(df, len(docs))
    print(f"已加载 {len(docs)} 篇文献, vocab={len(df)}")

    print("\n[train 自评]")
    evaluate_on_train(docs, idf)

    print("\n正在加载测试数据...")
    test = []
    with open('test.csv', 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            test.append({'ID': r['ID'], 'raw_input': r['raw_input']})
    if limit:
        test = test[:limit]
    print(f"已加载 {len(test)} 条测试数据")

    print("\n正在处理测试数据...")
    results = []
    label_counts = Counter()
    for i, item in enumerate(test):
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{len(test)}，耗时 {time.time()-start:.1f}s")
        stmt = item['raw_input']
        qset = set(tokenize(stmt))
        pd, ps, st = retrieve(qset, docs, idf)
        if pd is None:
            pd = next(iter(docs))
            ps = docs[pd]['sec_toks'][0][0] if docs[pd]['sec_toks'] else ''
            st = docs[pd]['sec_text'].get(ps, '')
        label = predict_label(stmt, st)
        label_counts[label] += 1
        results.append({'ID': item['ID'], 'review_id': pd,
                        'section_name': ps, 'label': label})

    print("\n正在保存结果...")
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ID', 'review_id', 'section_name', 'label'])
        w.writeheader()
        w.writerows(results)
    print(f"结果已保存到 {out}")
    print(f"标签分布: 0={label_counts[0]} 1={label_counts[1]}")
    print(f"总耗时: {time.time()-start:.1f}s")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只预测前N条test, 0=全部')
    ap.add_argument('--out', default='submission.csv', help='输出文件路径')
    a = ap.parse_args()
    main(a.limit, a.out)
