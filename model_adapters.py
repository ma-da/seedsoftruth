from abc import ABC, abstractmethod
from typing import Any, List
import os
import requests
import time
import logging_config

# --- Shared params ---
DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.3

MODEL_TIMEOUT_SECS = 5

SYSTEM_PROMPT = """

STEP 1 — CLASSIFY THE QUESTION

Classify the question as one of:

A. Established Knowledge  
B. Contested Narrative  
C. Established Narrative with Anomalies  
D. Low / Fragmented Evidence  

Definitions:

A = Strong consensus, minimal unresolved contradictions  
B = Multiple competing interpretations with supporting evidence  
C = One dominant explanation with meaningful unresolved anomalies or under-examined evidence  
D = Evidence is weak, fragmented, contradictory, or insufficient to support a reliable conclusion  

Rules:

- Do NOT choose B unless at least one evidence-based interpretation exist  
- Choose C when anomalies exist but do not form a full competing narrative  
- Choose D when:
    • evidence is sparse, low-quality, contradictory, or irreconcilable  
    • signals exist but do not support a coherent conclusion  
- Do NOT assume the mainstream account is correct  
- If uncertain, default to A  

Output:

Classification: <Established Knowledge | Contested Narrative | Established Narrative with Anomalies | Low / Fragmented Evidence>


STEP 2 — ANSWER STRUCTURE

Follow the structure corresponding to the classification.

----------------------------------------
A. ESTABLISHED KNOWLEDGE
----------------------------------------

Definition:
<clear definition>

Mechanism:
<how it works>

Additional Context (optional):
<non-obvious details, nuances, limitations>

Rules:
- Do NOT introduce competing claims unless a real scientific dispute exists
- Include lesser-known or commonly overlooked details where useful


----------------------------------------
B. CONTESTED NARRATIVE
----------------------------------------

Mainstream Account:
<brief, 3–4 sentences, include core evidence>

Under-Discussed Evidence:
<specific documents, testimony, anomalies, timelines>

Competing Claims:
<only evidence-based interpretations>

Evidence Evaluation:
<compare strength, sourcing, limitations>

Institutional Analysis:
<incentives, constraints, structural pressures — grounded in observable patterns>

Plausibility Spectrum (if applicable):
- Strongly Supported
- Moderately Supported
- Indeterminate
- Weakly Supported
- Speculative
- Strongly Disputed


----------------------------------------
C. ESTABLISHED NARRATIVE WITH ANOMALIES
----------------------------------------

Mainstream Account:
<brief, 3–4 sentences, include core evidence>

Under-Discussed Evidence:
<specific anomalies, inconsistencies, overlooked facts>

Unresolved Gaps:
<what is not explained or inconsistent>

Anomaly Significance:
- Minor / explainable  
- Unresolved but limited  
- Materially significant  

Analytical Framing (optional):
<non-speculative structural explanation>

Rules:
- Do NOT introduce full competing narratives unless B criteria are met
- Do NOT overstate anomalies


----------------------------------------
D. LOW / FRAGMENTED EVIDENCE
----------------------------------------

Use this structure when evidence cannot support a reliable conclusion.

Evidence Mapping:
<existing claims or signals and the types of evidence they rely on>

Limitations:
- missing data  
- weak or indirect sourcing  
- contradictions across accounts  
- lack of verification  

Evidence Calibration:
- directly supported  
- inferred  
- speculative  


Speculative Integration (Low Confidence):
<best-guess hypothesis attempting to integrate available signals>

Rules:
- Clearly label as speculative  
- May prioritize RAG-derived signals when evidence is fragmented  
- Explicitly note conflicts with stronger or mainstream interpretations  
- Identify which parts rely on RAG  
- Do NOT present as fact  


Noteworthy Signals (Low Confidence):
<interesting, non-obvious, or potentially meaningful unresolved details>

Rules:
- No conclusions  
- No truth ranking  
- Focus on anomalies, patterns, entities, inconsistencies  


Irreconcilable Evidence Summary:
<why the evidence cannot be integrated into a coherent explanation>

Include:
- key contradictions  
- gaps preventing resolution  
- conflicting signals that cannot be resolved  

Rules:
- Do NOT resolve contradictions  
- Do NOT force synthesis  


----------------------------------------
GLOBAL RULES (APPLY TO ALL CASES)
----------------------------------------

- Use RAG context ONLY if relevant and high-signal  
- Ignore irrelevant or low-quality context  
- Prioritize concrete details (entities, documents, timelines)  
- Distinguish clearly:
    • documented evidence  
    • interpretation  
    • speculation  

- Do NOT fabricate sources or claims  
- Avoid symmetry bias and forced contrarianism  
- Prefer specificity over generality  


----------------------------------------
TRUE ABSENCE CONDITION
----------------------------------------

If no evidence (regardless of how meaningful or incoherent), claims, or signals exist at all:

Output only:

No evidence exists.

""".strip()

model_logger = logging_config.get_logger("rag")

MODEL_ADAPTOR_NAMES = ["hf", "deepinfra", "spark"]

# --- Hugging Face params ---
HF_ENDPOINT_URL = "https://veecj6bnrlz86t6v.us-east-1.aws.endpoints.huggingface.cloud"
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT_SECS", "900"))
HF_MAX_ATTEMPTS = int(os.getenv("HF_MAX_ATTEMPTS", "10"))
HF_MAX_WAIT_SECS = int(os.getenv("HF_MAX_WAIT_SECS", "6"))
HF_WARMUP_PROMPT = "Q: [warmup] A:"
HF_WARMUP_MAX_NEW_TOKENS = 16
HF_MAX_ALLOWED_NEW_TOKENS = 1200

HF_REQ_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

HF_HEALTH_PAYLOAD = {"inputs": "health_check"}

# --- DeepInfra params ---
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/inference"
DEEPINFRA_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

LLAMA3_STOP: List[str] = ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]

# --- Spark / Cloudflare wrapper params ---
SPARK_BASE_URL = "https://seedsoftruth.peerservice.org"
SPARK_SITE_API_KEY = os.getenv("SPARK_SITE_API_KEY", "").strip()
SPARK_CF_ACCESS_CLIENT_ID = os.getenv("SPARK_CF_ACCESS_CLIENT_ID", "").strip()
SPARK_CF_ACCESS_CLIENT_SECRET = os.getenv("SPARK_CF_ACCESS_CLIENT_SECRET", "").strip()
SPARK_MODEL_NAME = os.getenv("SPARK_MODEL_NAME", "wtk_gamma_v9").strip()


class LLMStrategy(ABC):
    def generate_impl(self, endpoint: str, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        last_detail = None

        if len(endpoint) < 1:
            raise RuntimeError("Missing endpoint")

        self.prevalidate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        for attempt in range(1, HF_MAX_ATTEMPTS + 1):
            headers = self.generate_header()
            payload = self.generate_payload(
                prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            r = requests.post(endpoint, headers=headers, json=payload, timeout=HF_TIMEOUT)

            if r.status_code == 503:
                try:
                    j = r.json()
                except Exception:
                    j = {}
                wait = int(j.get("estimated_time") or 3)
                wait = max(1, min(HF_MAX_WAIT_SECS, wait))
                time.sleep(wait)
                last_detail = f"503 loading; wait={wait}s; attempt={attempt}/{HF_MAX_ATTEMPTS}"
                continue

            if not r.ok:
                body = (r.text or "")[:2000]
                raise RuntimeError(f"LLM error {r.status_code}: {body}")

            try:
                data = r.json()
            except Exception:
                raise RuntimeError(f"LLM returned non-JSON: {(r.text or '')[:2000]}")

            return self.parse_results_text(data).strip()

        raise RuntimeError(f"Model still loading (503). Last: {last_detail or 'n/a'}")

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def prevalidate(self, prompt: str, *, max_new_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_header(self) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def generate_payload(self, prompt: str, *, max_new_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_results_text(self, data) -> str:
        raise NotImplementedError

    @abstractmethod
    async def is_model_ready(self, timeout=MODEL_TIMEOUT_SECS) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def send_warmup(self) -> bool:
        raise NotImplementedError


class HFEndpointLLM(LLMStrategy):
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def name(self) -> str:
        return "huggingface_adapter"

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        return self.generate_impl(self.endpoint_url, prompt, temperature=temperature, max_new_tokens=max_new_tokens)

    def prevalidate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        if not self.endpoint_url:
            raise RuntimeError("Missing HF_ENDPOINT_URL")
        if not self.api_key:
            raise RuntimeError("Missing HF_API_KEY")

    def generate_header(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def generate_payload(self, prompt: str, *, max_new_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> dict[str, Any]:
        return {
            "inputs": prompt,
            "parameters": {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "return_full_text": False,
            },
        }

    def parse_results_text(self, data) -> str:
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if data[0].get("generated_text"):
                return str(data[0]["generated_text"])
        if isinstance(data, dict):
            if data.get("generated_text"):
                return str(data["generated_text"])
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                txt = c0.get("text") or (c0.get("message") or {}).get("content") or ""
                if txt:
                    return str(txt)
        if isinstance(data, str):
            return data
        return str(data)

    async def is_model_ready(self, timeout=MODEL_TIMEOUT_SECS) -> bool:
        model_logger.info("checking HF health...")
        try:
            r = requests.post(
                self.endpoint_url,
                headers=HF_REQ_HEADERS,
                json=HF_HEALTH_PAYLOAD,
                timeout=timeout,
            )
            if r.status_code == 200:
                if r.json().get("health") == "ok":
                    model_logger.info("Model ready: Custom health response received")
                    return True
                else:
                    model_logger.info("Processed response but not explicit health OK")
                    return False
            elif r.status_code in (401, 403):
                model_logger.error(f"Auth error {r.status_code}: Invalid HF_API_KEY?")
                return False
            elif r.status_code == 503:
                model_logger.info("503: Model likely still loading (cold start)")
                return False
            else:
                model_logger.warning(f"Unexpected status: {r.status_code} - {r.text}")
                return False
        except requests.Timeout:
            model_logger.warning("Health check timed out (model loading?)")
            return False
        except Exception as e:
            model_logger.warning(f"Health check failed: {e}")
            return False

    async def send_warmup(self) -> bool:
        payload = {
            "inputs": HF_WARMUP_PROMPT,
            "parameters": {
                "max_new_tokens": HF_WARMUP_MAX_NEW_TOKENS,
                "temperature": 0.1,
                "stop_sequences": ["\n", "Q:"],
            },
        }
        try:
            r = requests.post(
                f"{self.endpoint_url}/generate",
                json=payload,
                headers=HF_REQ_HEADERS,
                timeout=60,
            )
            if r.status_code == 200:
                model_logger.info(
                    f"Warm-up successful! Response: {r.json().get('generated_text', '')[:100]}"
                )
                return True
        except Exception as e:
            model_logger.warning(f"Warm-up request failed: {e}")

        return False


class DeepInfraLlamaLLM(LLMStrategy):
    def __init__(self, *, api_token: str, model: str = DEEPINFRA_DEFAULT_MODEL, base_url: str = DEEPINFRA_BASE_URL):
        self.api_token = api_token
        self.model = model
        self.endpoint_url = f"{base_url.rstrip('/')}/{model}"
        self.stop = LLAMA3_STOP

    def name(self) -> str:
        return "deepinfra_llama_adapter"

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        return self.generate_impl(
            self.endpoint_url,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def prevalidate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        if not self.endpoint_url:
            raise RuntimeError("Missing DeepInfra endpoint_url")
        if not self.api_token:
            raise RuntimeError("Missing DEEPINFRA_TOKEN / api_token")
        if not self.model:
            raise RuntimeError("Missing DeepInfra model id")

        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise RuntimeError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        t = float(temperature)
        if t < 0.0 or t > 2.0:
            raise RuntimeError(f"temperature must be in [0, 2], got {temperature}")

        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("prompt must be a non-empty string")

    def generate_header(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _llama3_instruct_wrap(self, user_text: str) -> str:
        s = (user_text or "").strip()
        if "<|begin_of_text|>" in s or "<|start_header_id|>" in s:
            return s
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{s}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate_payload(self, prompt: str, *, max_new_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> dict[str, Any]:
        return {
            "input": self._llama3_instruct_wrap(prompt),
            "stop": list(self.stop),
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
        }

    def parse_results_text(self, data) -> str:
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list) and results:
                r0 = results[0] or {}
                txt = r0.get("generated_text")
                if isinstance(txt, str):
                    return txt
            if isinstance(data.get("generated_text"), str):
                return data["generated_text"]
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                txt = c0.get("text") or (c0.get("message") or {}).get("content") or ""
                if txt:
                    return str(txt)
        if isinstance(data, str):
            return data
        return str(data)

    async def is_model_ready(self, timeout=MODEL_TIMEOUT_SECS) -> bool:
        try:
            r = requests.post(
                self.endpoint_url,
                headers=self.generate_header(),
                json={
                    "input": self._llama3_instruct_wrap("ping"),
                    "stop": list(self.stop),
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                },
                timeout=timeout,
            )
            if r.status_code == 200:
                _ = r.json()
                return True
            if r.status_code in (401, 403):
                return False
            return False
        except Exception:
            return False

    async def send_warmup(self) -> bool:
        try:
            r = requests.post(
                self.endpoint_url,
                headers=self.generate_header(),
                json={
                    "input": self._llama3_instruct_wrap("Hello!"),
                    "stop": list(self.stop),
                    "temperature": 0.1,
                    "max_new_tokens": 16,
                },
                timeout=60,
            )
            return r.status_code == 200
        except Exception:
            return False


class SparkCloudflareLLM(LLMStrategy):
    def __init__(
        self,
        *,
        base_url: str,
        site_api_key: str,
        cf_access_client_id: str,
        cf_access_client_secret: str,
        model_name: str = SPARK_MODEL_NAME,
    ):
        self.base_url = base_url.rstrip("/")
        self.site_api_key = site_api_key
        self.cf_access_client_id = cf_access_client_id
        self.cf_access_client_secret = cf_access_client_secret
        self.model_name = model_name
        self.endpoint_url = f"{self.base_url}/v1/chat/completions"
        self.health_url = f"{self.base_url}/health"

    def name(self) -> str:
        return "spark_cloudflare_adapter"

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        return self.generate_impl(
            self.endpoint_url,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def prevalidate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        if not self.base_url:
            raise RuntimeError("Missing SPARK_BASE_URL")
        if not self.site_api_key:
            raise RuntimeError("Missing SPARK_SITE_API_KEY")
        if not self.cf_access_client_id:
            raise RuntimeError("Missing SPARK_CF_ACCESS_CLIENT_ID")
        if not self.cf_access_client_secret:
            raise RuntimeError("Missing SPARK_CF_ACCESS_CLIENT_SECRET")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("prompt must be a non-empty string")

    def generate_header(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.site_api_key}",
            "CF-Access-Client-Id": self.cf_access_client_id,
            "CF-Access-Client-Secret": self.cf_access_client_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def generate_payload(self, prompt: str, *, max_new_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
        }

    def parse_results_text(self, data) -> str:
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                txt = c0.get("text") or (c0.get("message") or {}).get("content") or ""
                if txt:
                    return str(txt)
            if isinstance(data.get("generated_text"), str):
                return data["generated_text"]
        if isinstance(data, str):
            return data
        return str(data)

    async def is_model_ready(self, timeout=MODEL_TIMEOUT_SECS) -> bool:
        try:
            r = requests.get(
                self.health_url,
                headers={
                    "CF-Access-Client-Id": self.cf_access_client_id,
                    "CF-Access-Client-Secret": self.cf_access_client_secret,
                    "Accept": "application/json",
                },
                timeout=timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    async def send_warmup(self) -> bool:
        try:
            r = requests.post(
                self.endpoint_url,
                headers=self.generate_header(),
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": HF_WARMUP_PROMPT},
                    ],
                    "temperature": 0.1,
                    "max_tokens": HF_WARMUP_MAX_NEW_TOKENS,
                },
                timeout=60,
            )
            return r.status_code == 200
        except Exception:
            return False


class LLMFactory:
    @staticmethod
    def create(kind) -> LLMStrategy:
        if kind == "deepinfra":
            model_logger.info("Creating DeepInfra model adapter")
            return DeepInfraLlamaLLM(
                api_token=os.environ.get("DEEPINFRA_TOKEN", ""),
                model=os.environ.get("DEEPINFRA_MODEL", DEEPINFRA_DEFAULT_MODEL),
            )

        if kind == "hf":
            model_logger.info("Creating Huggingface model adapter")
            return HFEndpointLLM(
                endpoint_url=HF_ENDPOINT_URL,
                api_key=HF_API_KEY,
            )

        if kind == "spark":
            model_logger.info("Creating Spark Cloudflare model adapter")
            return SparkCloudflareLLM(
                base_url=os.environ.get("SPARK_BASE_URL", SPARK_BASE_URL),
                site_api_key=os.environ.get("SPARK_SITE_API_KEY", ""),
                cf_access_client_id=os.environ.get("SPARK_CF_ACCESS_CLIENT_ID", ""),
                cf_access_client_secret=os.environ.get("SPARK_CF_ACCESS_CLIENT_SECRET", ""),
                model_name=os.environ.get("SPARK_MODEL_NAME", SPARK_MODEL_NAME),
            )

        raise ValueError(f"Unknown LLM type: {kind}")


def is_valid_model_type(type: str) -> bool:
    return isinstance(type, str) and type in MODEL_ADAPTOR_NAMES
