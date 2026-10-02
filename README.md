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

## 实测结果

- **检索**：train 全量 77.6%（训练100/101-200 两批 82%/72%）。金标文献有 **8% 不在语料中**，词面匹配上限≈92%；已测试 top2聚合/联合argmax/bigram/词干等 10 种变体，均无提升，词袋法到顶。想再提分只能上语义向量检索。
- **LLM 标签**：检索正确子集 97-100% 正确；端到端（Kaggle 口径）**70-71%**。
- **提交**：前 500 条 100% LLM 判定，label 分布 0=278 / 1=222。

## 模型管线（全部免费，令牌走环境变量）

```
1. NIM nemotron-3-super-120b-a12b   主力, ~2s/条 (ultra-550b/253b 已于 8/30 404下架)
2. freellm auto                     次选, 3-8s/条, 服务不稳时抖动
3. freellm gpt-oss-20b              备胎, 质量≈4o Mini
4. openrouter nemotron-3-ultra:free 备份, 额度恢复自动生效
5. freellm gpt-oss-120b             兜底, 慢(~50s/条)
```

- Gemini 项目级 403（需结算）；GitHub Models 退役 brownout；HuggingFace 网络不可达——均已移除。
- freellm 曾拒绝 `max_tokens=2048`（≤512 正常）→ 管线统一 256/512。

## 环境变量（setx 持久化，代码零硬编码）

`NVIDIA_API_KEY` / `FREELLM_API_KEY` / `OPENROUTER_API_KEY`

## 用法

```bash
# 纯检索+pattern 全量 test（无API, ~3分钟）
python rag_fact_verify.py

# LLM 流水线: train 前100条审计 / test 前500条
python rag_run_audit.py --train 0 100
python rag_run_audit.py --limit 500 --workers 4 --out submission_test.csv
```

## 踩坑记录

1. **NIM 模型下架**：`/models` 列表有缓存，推理 404 时直接换同系模型并实测。
2. **ckpt 每 25 条才落盘**（`(i+1)%25`）：小批量任务 ckpt 为空属正常。
3. **慢尾**：NIM 偶发 60s+ 甚至 120s 超时，管线靠多级兜底消化，勿在单条上死等。
4. **解析**：judge 取响应第一个 `[01]`；SYSTEM_PROMPT 要求只输出单字符，换 reasoning 模型前必须先验证输出不带推理回显。
