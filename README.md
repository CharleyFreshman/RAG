# 基于文献的事实验证系统 — 技术报告

**任务**：给定文献库 original_data.csv（QASPER），对 test 中每条陈述输出
`review_id + section_name + label(1真/0伪)`。评分：review_id+section_name 全对才计检索分，再看 label。

## 架构

```
test 陈述 ──> 检索 (rag_fact_verify.py): IDF词面打分, 文献argmax → 段落argmax
         ──> 判定 (rag_llm_api.py + rag_run_audit.py): LLM 只输出 1/0
              └─ LLM 失败(重试1次仍败) → pattern 规则回退
         ──> submission_test.csv
```

## 核心文件

| 文件 | 职责 |
|---|---|
| `rag_fact_verify.py` | 语料加载/IDF/检索/pattern标签；支持 `--limit/--out` |
| `rag_llm_api.py` | 多平台模型管线 `OpinionAnalyzer`，禁止跨平台 fallback |
| `rag_run_audit.py` | 流水线（检索→LLM→审计），`run()` 并行 + 明细落盘 |

## 技术细节

### 数据格式

- `original_data.csv`：`id, title, abstract, full_text`；`full_text` 是 Python dict 字面量字符串，`ast.literal_eval` 解析为 `{section_name: [...], paragraphs: [[...]]}`，逐节配对。
- `train.csv`：`ans, review_id, section_name, label`（金标三元组）；`test.csv`：`ID, raw_input`；提交格式 `ID, review_id, section_name, label`。

### 检索（rag_fact_verify.py）

- **分词**：`[a-z]+` 正则、小写、长度>2、去 sklearn 英文停用词。
- **IDF**：`log((N+1)/(df+1)) + 1`，df 按"词是否出现在该文档"统计。
- **打分（V5）**：文献级 `0.5·doc覆盖 + 1.0·max段落覆盖 + 0.5·title+abstract覆盖`，覆盖 = 查询词命中该索引的 IDF 之和；argmax 选文献，文献内按段落覆盖 argmax 选 section。
- **无命中兜底**：返回 None 时取语料首篇首节，保证提交行完整。

### Pattern 回退标签（LLM 失败时）

四个失真信号加权：数字缺失 −3、否定错配 −2、反转结构（rather than/instead of）未现 −3、量词夸大 −3；`score ≥ −2` 判 1，否则 0；检索文本为空强势判 0。train 网格搜索调参，label acc≈0.63。

### LLM 判定（rag_run_audit.py + rag_llm_api.py）

- **Prompt**：SYSTEM 规则（1=忠实转述 / 0=矛盾·夸大·改数字量词否定）+ 段落原文（`--max-sec-chars` 默认 3500）+ 陈述；要求只输出单字符，解析取响应第一个 `[01]`。
- **辅助提取**（默认开）：先用 `auto` 模型抽取陈述相关关键句拼进主 prompt；该平台停用后静默返回空串，不影响主判定。
- **容错**：调用经 64 线程池提交，`result(timeout=150)` 防挂起；失败自动重试一次，仍失败回退 pattern（明细 `src` 列区分）；NIM 偶发 `content=None` 已加 `(content or '')` 保护。
- **管线**：`MODEL_PIPELINE` 每项绑定唯一平台，禁止跨平台 fallback，防免费额度被错误平台消耗；参数统一 `temp=0.2, max_tokens=256/512`（freellm 曾拒绝 2048）。

### 续跑与并行

- **追加续跑**（默认）：`_count_done` 数 `submission_test.csv` 已有行数，从进度后预测 `--next N`（0=剩余全部），防重复烧额度；`--fresh` 忽略进度全量重跑。
- **checkpoint**：并行结果每 25 条落盘 `ckpt_tmp.csv`（`.gitignore` 排除），正常结束即删除；train 审计先存明细 CSV 再统计，防统计崩溃丢结果。
- **并行**：`--workers` 控制并发；明细含 `t_retr/t_llm/model/src/pchars` 供事后审计。

## 实测结果（train，重试机制生效后重跑）

| 批次 | 检索 | 标签·检索正确前提 | 端到端(Kaggle口径) |
|---|---|---|---|
| 1-100 | 82% | 93.9% | 77% |
| 101-200 | 73% | 94.5% | 69% |
| 201-300 | 83% | 89.2% | 74% |
| 301-400 | 76% | 93.4% | 71% |
| **1-400 合并** | **78.5%** | **92.7%** | **72.8%** |

- 检索是确定性词面匹配，全量 1000 条统计 77.6%。金标 8% 文献不在语料中（词面上限≈92%）；已测试 top2聚合/联合argmax/bigram/词干等 10 种变体均无提升，词袋到顶，再提分需语义向量检索。
- 错例大头是检索 MISS（词面信号弱，如公式占位段落），其次 LLM 判错；pattern 回退 44/400=11% 贡献少量错误。
- 提交：前 500 条 100% LLM 判定，label 分布 0=278 / 1=222。

## 模型管线（全部免费，令牌走环境变量）

```
生效:
1. NIM nemotron-3-super-120b-a12b  主力, ~2s/条 (ultra-550b/253b 已于 8/30 404下架)
2. openrouter nemotron-3-ultra:free 备用, 额度恢复自动生效

已停用(代码注释保留, 随时可恢复):
- mistral mistral-small-latest  免费层持续 429 (code 1300), 实际 0 判定
- gemini-3.6-flash  免费层要求绑定结算, 两轮尝试后 9/25 弃用
- freellm auto/gpt-oss-20b  上游 provider key 失效 (503 no_model_available)
- GitHub Models 退役 brownout; HuggingFace 网络不可达 —— 均已移除
```

- **判定承担统计**（train 1-400）：NIM super-120b 314 条 (78.5%) / OpenRouter ultra:free 42 条 (10.5%) / pattern 回退 44 条 (11%)。

## 环境变量（setx 持久化，代码零硬编码）

`NVIDIA_API_KEY` / `MISTRAL_API_KEY` / `FREELLM_API_KEY` / `OPENROUTER_API_KEY`

## 用法

```bash
# 纯检索+pattern 全量 test（无API, ~3分钟）
python rag_fact_verify.py

# LLM 流水线: train 前100条审计
python rag_run_audit.py --eval-train 100 --workers 4

# test 预测: 默认追加续跑(从 submission_test.csv 已有进度往后), --next 0 追完剩余
python rag_run_audit.py --next 0 --workers 4

# 全新全量重跑(忽略已有进度)
python rag_run_audit.py --fresh --workers 4
```

## 踩坑记录

1. **NIM 模型下架**：`/models` 列表有缓存，推理 404 时直接换同系模型并实测。
2. **ckpt 每 25 条才落盘**（`(i+1)%25`）：小批量任务 ckpt 为空属正常。
3. **平台静默跳过**：环境变量缺失 → client=None → 被 `except: continue` 吞掉，流水线无提示地跳过该模型。排查平台为何没被调用时，先查环境变量再查代理（诊断脚本: `test_freellm.py` / `test_gemini.py`）。
4. **解析**：judge 取响应第一个 `[01]`；换 reasoning 模型前必须先验证输出不带推理回显。
