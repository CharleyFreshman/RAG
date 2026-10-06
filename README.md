# 基于文献的事实验证系统 — 技术报告

**任务**：给定文献库 original_data.csv（QASPER），对 test 中每条陈述输出
`review_id + section_name + label(1真/0伪)`。评分先查检索对不对，再看标签对不对。

## 架构

```
test 陈述 ──> 检索: IDF模式匹配 (rag_fact_verify.py)
              ├─ 文献级打分: 0.5·doc覆盖 + 1.0·max段落覆盖 + 0.5·title+abstract覆盖
              └─ 段落级: 最优段argmax
         ──> 判定: LLM (rag_llm_api.py + rag_run_audit.py)
              ├─ 段落原文(≤3500字符) + 陈述 → 只输出 1/0
              └─ 失败自动回退 pattern 规则(数字/否定/反转/量词失真检测)
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

- `original_data.csv`（QASPER 文献库）：`id, title, abstract, full_text`；`full_text` 是 Python dict 字面量字符串，用 `ast.literal_eval` 解析为 `{section_name: [...], paragraphs: [[...]]}`，逐节配对。
- `train.csv`：`ans(陈述), review_id, section_name, label`（金标三元组）；`test.csv`：`ID, raw_input`。
- 提交格式：`ID, review_id, section_name, label`。评分口径：review_id+section_name 全对才计检索分，再看 label。

### 检索（rag_fact_verify.py）

- **分词**：`[a-z]+` 正则、小写、长度>2、去 sklearn 英文停用词。
- **IDF**：`log((N+1)/(df+1)) + 1`，df 按"词是否出现在该文档（title+abstract+全文）"统计。
- **文献级打分**（V5）：`score = 0.5·doc覆盖 + 1.0·max(段落覆盖) + 0.5·title+abstract覆盖`，覆盖 = 查询词集中命中该索引的词的 IDF 之和；argmax 选文献，再在文献内按段落覆盖 argmax 选 section。
- **无命中兜底**：`retrieve` 返回 None 时取语料首篇首节（保证提交行完整）。

### Pattern 规则标签（LLM 失败时的回退）

四个失真信号加权：数字缺失 −3、否定错配 −2、反转结构（rather than/instead of）未现 −3、量词夸大（all/only/most…陈述有而原文无）−3；`score ≥ −2` 判 1，否则判 0；检索文本为空强势判 0。train 网格搜索调参，label acc≈0.63。

### LLM 判定（rag_run_audit.py + rag_llm_api.py）

- **Prompt**：SYSTEM 规则（1=忠实转述 / 0=矛盾·夸大·改数字量词否定）+ 段落原文（截断 `--max-sec-chars`，默认 3500 字符）+ 陈述；要求只输出单字符，解析取响应中第一个 `[01]`。
- **辅助提取**（可选，默认开）：先用 `auto` 模型抽取与陈述相关的关键句拼进主 prompt；freellm 平台停用后该步静默返回空串，不影响主判定。
- **硬超时**：每次调用经 64 线程 `_HARD_POOL` 提交，`result(timeout=100)` 防单条挂起；超时/无 `[01]` → 回退 pattern 标签（`src` 列记录 `llm`/`pattern`）。
- **管线配置**：`MODEL_PIPELINE` 每项绑定唯一平台（`preferred`），`_resolve_framework` 只返回该平台——**禁止跨平台 fallback**，防止免费额度被错误平台消耗。参数：nemotron-super `temp=0.2, max_tokens=512`；openrouter 备份 `256`。

### 续跑与容错

- **追加续跑**（默认）：`_count_done` 数 `submission_test.csv` 已有行数，从该行号往后预测 `--next N`（0=剩余全部），防重复烧额度；`--fresh` 忽略进度全量重跑。
- **checkpoint**：并行结果每 25 条落盘 `ckpt_tmp.csv`（`.gitignore` 排除），正常结束即删除；train 审计先存明细 CSV 再统计，防统计崩溃丢结果。
- **并行**：`--workers` 控制 `ThreadPoolExecutor` 并发，明细含 `t_retr/t_llm/model/src/pchars` 供事后审计。

## 实测结果

- **检索**：train 全量 77.6%（重跑分批: 1-100 82% / 101-200 73% / 201-300 83% / 301-400 76%，1-400 合并 78.5%）。金标文献有 **8% 不在语料中**，词面匹配上限≈92%；已测试 top2聚合/联合argmax/bigram/词干等 10 种变体，均无提升，词袋法到顶。想再提分只能上语义向量检索。
- **LLM 标签**：检索正确子集 89-95% 正确（重跑: 1-100 77/82=93.9%、101-200 69/73=94.5%、201-300 89.2%、301-400 93.4%）；端到端（Kaggle 口径）**69-77%**（重跑: 77%/69%/74%/71%，1-400 合并≈72.8%）。失败重试一次后回退率 17→10/100。
- **提交**：前 500 条 100% LLM 判定，label 分布 0=278 / 1=222。

## 模型管线（全部免费，令牌走环境变量）

```
生效:
1. NIM nemotron-3-super-120b-a12b  主力, ~2s/条 (ultra-550b/253b 已于 8/30 404下架)
2. openrouter nemotron-3-ultra:free 备用, 额度恢复自动生效

已停用(代码中注释保留, 随时可恢复):
- mistral mistral-small-latest    备用管线; key 已配置, 但免费层持续 429 限流(code 1300), 实际 0 判定, 额度刷新后自动生效(快速失败仅加一次跳转)
                                   
- gemini-3.6-flash   OpenAI兼容端点实测要求绑定结算(billing), 免费层不可用, 9/25 注释
- freellm auto/gpt-oss-20b  代理在线但上游 provider key 全部失效(503 no_model_available), dashboard 补 key 后可恢复
```

- **判定承担统计**（train 1-400 重跑）：NIM super-120b 314 条 (78.5%) / OpenRouter ultra:free 42 条 (10.5%) / pattern 回退 44 条 (11.0%) / mistral 0 条（429）。
- Gemini 两轮尝试均失败: 项目级 403（需结算）、OpenAI 兼容端点同样要求结算——为 benchmark 不值得绑卡, 彻底放弃。
- GitHub Models 退役 brownout；HuggingFace 网络不可达——均已移除。
- freellm 曾拒绝 `max_tokens=2048`（≤512 正常）→ 管线统一 256/512。

## 环境变量（setx 持久化，代码零硬编码）

`NVIDIA_API_KEY` / `MISTRAL_API_KEY`（已配置，免费层 429 限流中）/ `FREELLM_API_KEY`（代理停用, 暂闲置）/ `OPENROUTER_API_KEY`
（`GEMINI_API_KEY` 无需配置——Gemini 已因结算要求弃用）

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
3. **慢尾**：NIM 偶发 60s+ 超时，管线靠多级兜底消化，勿在单条上死等。
4. **解析**：judge 取响应第一个 `[01]`；SYSTEM_PROMPT 要求只输出单字符，换 reasoning 模型前必须先验证输出不带推理回显。
5. **平台静默跳过**：环境变量缺失 → client=None → "无可用客户端"被 `except: continue` 吞掉，流水线无任何提示地跳过该模型。排查某平台为何没被调用时，先查环境变量再查代理（诊断脚本: `test_freellm.py` / `test_gemini.py`）。
6. **content=None**：NIM 偶发返回 `success=True` 但 `content=None`，解析处已加 `(content or '')` 保护，该条按失败回退 pattern。
7. **Mistral 免费 429**：key 有效（`models.list` 正常）但 chat 全部 429 code 1300，间隔重试无效——非每秒节流，是免费层硬配额，按月刷新；管线中快速失败自动落到 NIM，无需改码。
