"""
基于大模型的事实验证系统: NVIDIA NIM(主力) + FreeLLM(次选) + OpenRouter(额度恢复后自动生效)。
所有令牌一律从环境变量读取, 代码中不硬编码:
  NVIDIA_API_KEY / FREELLM_API_KEY / OPENROUTER_API_KEY
"""
from openai import OpenAI
import os
import time
from typing import Dict, List, Optional, Tuple, Union


def _env_key(name: str) -> str:
    """读取令牌环境变量, 未设置返回空串"""
    return os.environ.get(name, "").strip()


SYSTEM_PROMPT = """You are a fact verification assistant. Based ONLY on the given passage from a scientific paper, judge whether the statement is true or false.
Rules:
- 1 = the statement is consistent with the passage (correct paraphrase, faithful summary)
- 0 = the statement contradicts, exaggerates, over-generalizes, or alters numbers/quantifiers/negation compared with the passage
Output ONLY a single digit (1 or 0), nothing else."""


class OpinionAnalyzer:
    def __init__(self, use_freeflow: bool = False):
        self.use_freeflow = use_freeflow
        self.clients = {}
        self.freeflow_client = None

        if use_freeflow:
            try:
                from freeflow_llm import FreeFlow
                self.freeflow_client = FreeFlow()
            except ImportError:
                self.use_freeflow = False

        if not self.use_freeflow:
            self._init_standard_clients()

    def _init_standard_clients(self):
        nim_key = _env_key("NVIDIA_API_KEY")
        self.clients["nim"] = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nim_key,
            timeout=150,
            max_retries=1,
        ) if nim_key else None

        or_key = _env_key("OPENROUTER_API_KEY")
        self.clients["openrouter"] = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
            timeout=150,
            max_retries=1,
            default_headers={
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "FactAnalysis"
            }
        ) if or_key else None

        mistral_key = _env_key("MISTRAL_API_KEY")
        self.clients["mistral"] = OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=mistral_key,
            timeout=150,
            max_retries=1,
        ) if mistral_key else None

        # gemini_key = _env_key("GEMINI_API_KEY")
        # self.clients["gemini"] = OpenAI(
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        #     api_key=gemini_key,
        #     timeout=150,
        #     max_retries=1,
        # ) if gemini_key else None

        # fl_key = _env_key("FREELLM_API_KEY") or "EMPTY"  # 本地代理不校验, EMPTY仅为满足SDK非空要求
        # self.clients["freellm"] = OpenAI(
        #     base_url="http://localhost:3000/v1",
        #     api_key=fl_key,
        #     timeout=60,
        #     max_retries=1,
        # ) if fl_key else None


    def _resolve_framework(self, model_cfg: Dict) -> List[str]:
        preferred = model_cfg.get("preferred", "freellm")
        return [preferred] if self.clients.get(preferred) is not None else []

    def _call_standard(self, model_cfg: Dict, prompt: str) -> Tuple[bool, Optional[str], Optional[str]]:
        available_frameworks = self._resolve_framework(model_cfg)
        if not available_frameworks:
            return False, None, "无可用客户端"

        for framework in available_frameworks:
            cli = self.clients[framework]
            try:
                params = {
                    "model": model_cfg["id"],
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": model_cfg.get("max_tokens", 512),
                    "temperature": model_cfg.get("temp", 0.3),
                }

                if framework == "openrouter" and "extra" in model_cfg:
                    params["extra_body"] = model_cfg["extra"]

                resp = cli.chat.completions.create(**params)
                return True, resp.choices[0].message.content, None

            except Exception:
                continue

        return False, None, f"所有框架尝试失败: {available_frameworks}"

    def _call_freeflow(self, prompt: str) -> Tuple[bool, Optional[str], Optional[str]]:
        if not self.freeflow_client:
            return False, None, "FreeFlow 客户端未初始化"

        try:
            response = self.freeflow_client.chat.completions.create(
                model="auto",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return True, response.choices[0].message.content, None
        except Exception as e:
            return False, None, f"FreeFlow 调用失败: {str(e)[:100]}"

    def analyze(self, prompt: str, model_id: Optional[str] = None) -> Dict:
        if self.use_freeflow:
            success, content, error = self._call_freeflow(prompt)
            if success:
                return {"success": True, "model": "freeflow-auto", "content": content}
            return {"success": False, "error": error}

        # 主管线: MODEL_PIPELINE 顺序兜底 (各模型只走指定的平台)
        models = MODEL_PIPELINE if model_id is None else [
            cfg for cfg in MODEL_PIPELINE if cfg["id"] == model_id
        ]
        for cfg in models:
            success, content, error = self._call_standard(cfg, prompt)
            if success:
                return {"success": True, "model": cfg["id"], "content": content}

        return {"success": False, "error": "所有模型调用失败"}

    def analyze_batch(self, prompts: List[str], batch_size: int = 10, delay: float = 60.0) -> List[Dict]:
        """批量分析。免费平台有额度限制, 默认批间延迟60s防限流封禁"""
        results = []
        total = len(prompts)

        for i in range(0, total, batch_size):
            batch = prompts[i:i+batch_size]

            if len(batch) == 1:
                results.append(self.analyze(batch[0]))
            else:
                combined_prompt = "please analyze the following opinions separately, and label each result with 0/1:\n\n"
                for idx, opinion in enumerate(batch, 1):
                    combined_prompt += f"{idx}. {opinion}\n"

                result = self.analyze(combined_prompt)
                if result["success"]:
                    content = result["content"]
                    parts = content.split("\n\n")
                    parsed_results = []
                    for part in parts:
                        if part.strip():
                            parsed_results.append({
                                "success": True,
                                "model": result["model"],
                                "content": part.strip()
                            })

                    while len(parsed_results) < len(batch):
                        parsed_results.append({"success": False, "error": "解析失败, 结果数量不匹配"})

                    results.extend(parsed_results[:len(batch)])
                else:
                    for opinion in batch:
                        results.append(self.analyze(opinion))

            if i + batch_size < total:
                time.sleep(delay)

        return results

MODEL_PIPELINE: List[Dict] = [
    {
        "id": "nvidia/nemotron-3-super-120b-a12b",
        "preferred": "nim",
        "reason": "主力: 推理质量好, 实测2s/条; ultra-550b/253b 8月30日下架",
        "max_tokens": 512,
        "temp": 0.2
    },
    {
        "id": "mistral-small-latest",
        "preferred": "mistral",
        "reason": "Mistral 免费额度备用",
        "max_tokens": 512,
        "temp": 0.2
    },
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "preferred": "openrouter",
        "reason": "备份: 推理质量最好，免费额度50次/天恢复后自动生效",
        "max_tokens": 256,
        "temp": 0.2
    },
    # {
    #     "id": "gemini-3.6-flash",
    #     "preferred": "gemini",
    #     "reason": "Gemini 免费额度备用",
    #     "max_tokens": 512,
    #     "temp": 0.2
    # },
    # {
    #     "id": "auto",
    #     "preferred": "freellm",
    #     "reason": "次选: 实测最快(3-8s/条); 服务端自动路由, 无法指定模型",
    #     "max_tokens": 512,
    #     "temp": 0.2
    # },
    # {
    #     "id": "openai/gpt-oss-20b",
    #     "preferred": "freellm",
    #     "reason": "次选备胎: 质量≈4o Mini, 速度与 auto 打平",
    #     "max_tokens": 512,
    #     "temp": 0.3
    # },
]

if __name__ == "__main__":
    analyzer = OpinionAnalyzer(use_freeflow=False)
    opinion = '"人工智能模型通过增加参数规模就能无限提升推理能力。"'
    result = analyzer.analyze(opinion)

    if result["success"]:
        print(f"模型: {result['model']}")
        print(result["content"])
    else:
        print(f"错误: {result.get('error', '未知错误')}")
