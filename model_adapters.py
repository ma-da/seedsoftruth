from abc import ABC, abstractmethod
from typing import Protocol, Optional
import os
import requests
import time
import logging_config

# --- Shared params ---
DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.3

SYSTEM_PROMPT = """
Answer the question in 1–3 concise paragraphs (total <300 words).
Use proper spelling, punctuation, and spacing.
Do not run words together.
Avoid long strings of numbers.
NO LISTS. If unavoidable, then limit lists to 3 items maximum.
Focus only on the question asked, avoiding unrelated topics or meta-text (e.g., "Note:", "click here").
Do not suggest other references, "further reading", or "Note:" for the reader to explore, view, or to learn more.
Stop after the answer.
/no_think
""".strip()

model_logger = logging_config.get_logger("rag")

# --- Huggingface params ---

# endpoint wtk-trineday-mini-llama3-70b-muu
HF_ENDPOINT_URL = "https://d6pfgv6yisy4pld2.us-east-1.aws.endpoints.huggingface.cloud" #os.getenv("HF_ENDPOINT_URL", "").strip()

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
MODEL_TIMEOUT_SECS = 5


# -- Base LLM strategy --
# This class is the base class which contains common methods for all strategies.
# It contains hooks which should be overridden by implementations.
class LLMStrategy(ABC):

    # --- Common Template ---
    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        last_detail = None

        self.prevalidate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        for attempt in range(1, HF_MAX_ATTEMPTS + 1):
            headers = self.generate_header()
            payload = self.generate_payload(prompt, temperature=temperature)

            r = requests.post(HF_ENDPOINT_URL, headers=headers, json=payload, timeout=HF_TIMEOUT)

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
                raise RuntimeError(f"HF error {r.status_code}: {body}")

            try:
                data = r.json()
            except Exception:
                raise RuntimeError(f"HF returned non-JSON: {(r.text or '')[:2000]}")

            return self.parse_results_text(data).strip()

        raise RuntimeError(f"HF model still loading (503). Last: {last_detail or 'n/a'}")

    # --- Custom hooks to be customized per strategy --

    # Does prevalidation of params
    @abstractmethod
    def prevalidate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        raise NotImplementedError

    # Generates the header to be sent out with llm request
    @abstractmethod
    def generate_header(
            self,
    ) -> dict[str, str]:
        raise NotImplementedError

    # Generates the payload to be sent out with llm request
    @abstractmethod
    def generate_payload(
            self,
            prompt: str,
            *,
            max_new_tokens: int = DEFAULT_MAX_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE
    ) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def parse_results_text(
        self,
        data) -> str:
        raise NotImplementedError

    @abstractmethod
    async def is_model_ready(
        self,
        timeout=MODEL_TIMEOUT_SECS) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def send_warmup(
        self,
    ) -> bool:
        raise NotImplementedError


# --- HF LLM strategy --
class HFEndpointLLM(LLMStrategy):
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def prevalidate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        if not self.endpoint_url:
            raise RuntimeError("Missing HF_ENDPOINT_URL")
        if not self.api_key:
            raise RuntimeError("Missing HF_API_KEY")

    # Generates the header to be sent out with llm request
    def generate_header(
            self,
    ) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # Generates the payload to be sent out with llm request
    def generate_payload(
            self,
            prompt: str,
            *,
            max_new_tokens: int = DEFAULT_MAX_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE
    ) -> dict[str, str]:
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
        model_logger.info("checking health...")
        try:
            r = requests.post(f"{self.endpoint_url}",
                              headers=HF_REQ_HEADERS,
                              json=HF_HEALTH_PAYLOAD,
                              timeout=timeout)
            if r.status_code == 200:
                if r.json().get("health") == "ok":
                    model_logger.info("Model ready: Custom health response received")
                    return True
                else:
                    model_logger.info("Processed response but not explicit health OK")
                    return False  # Or False if strict
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
                "stop_sequences": ["\n", "Q:"]
            }
        }
        try:
            r = requests.post(
                f"{HF_ENDPOINT_URL}/generate",
                json=payload,
                headers=HF_REQ_HEADERS,
                timeout=60
            )
            if r.status_code == 200:
                model_logger.info(f"Warm-up successful! Response: {r.json().get('generated_text', '')[:100]}")
                return True
        except Exception as e:
            model_logger.warning(f"Warm-up request failed: {e}")

        return False


# -- LLM Factory builder --
class LLMFactory:
    @staticmethod
    def create(kind) -> LLMStrategy:
        if kind == "hf":
            return HFEndpointLLM(
                endpoint_url=HF_ENDPOINT_URL,
                api_key=HF_API_KEY,
            )

        raise ValueError(f"Unknown LLM type: {kind}")