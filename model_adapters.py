from abc import ABC, abstractmethod
from typing import Protocol, Optional, Any, List
import os
import requests
import time
import logging_config

# --- Shared params ---
DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.3

MODEL_TIMEOUT_SECS = 5

#SYSTEM_PROMPT = """
#Answer the question in 1–3 concise paragraphs (total <300 words).
#Use proper spelling, punctuation, and spacing.
#Do not run words together.
#Avoid long strings of numbers.
#NO LISTS. If unavoidable, then limit lists to 3 items maximum.
#Focus only on the question asked, avoiding unrelated topics or meta-text (e.g., "Note:", "click here").
#Do not suggest other references, "further reading", or "Note:" for the reader to explore, view, or to learn more.
#Stop after the answer.
#/no_think
#""".strip()

#SYSTEM_PROMPT = """
#You are a forensic analyst of historical and political narratives.
#
#Your task is to:
#1. Present the official account accurately.
#2. Identify documented contradictions.
#3. Evaluate the evidentiary strength of each.
#4. Analyze structural incentives without assuming coordination.
#5. Clearly distinguish evidence from speculation.
#
#Do not assert unverified claims as fact.
#Do not imply coordination without strong documentation.
#End with a plausibility spectrum assessment.
#""".strip()

#SYSTEM_PROMPT = """
#You produce structured analytical responses.
#
#First determine whether the question concerns:
#
#1. Established knowledge or factual explanations.
#2. Contested narratives involving competing interpretations.
#
#If the topic is established knowledge:
#    Provide a clear explanatory synthesis organized into sections.
#
#If the topic is contested:
#    Present the mainstream account, competing claims, evaluate evidence, analyze incentives, and end with a Plausibility Spectrum.
#
#Never manufacture controversy where none exists.
#Clearly distinguish evidence from speculation.
#If fewer than two credible interpretations exist, treat the question as established knowledge.
#"""

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

"""

model_logger = logging_config.get_logger("rag")

MODEL_ADAPTOR_NAMES = ["hf", "deepinfra"]


# --- Huggingface params ---

# endpoint wtk-trineday-mini-llama3-70b-muu
#HF_ENDPOINT_URL = "https://d6pfgv6yisy4pld2.us-east-1.aws.endpoints.huggingface.cloud" #os.getenv("HF_ENDPOINT_URL", "").strip()

# wtk-gamma-llama3-70b-v9
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


# -- Deep Infra params --

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/inference"      # should end with slash
DEEPINFRA_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

LLAMA3_STOP: List[str] = ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]


# -- Base LLM strategy --

# This class is the base class which contains common methods for all strategies.
# It contains hooks which should be overridden by implementations.
class LLMStrategy(ABC):

    # --- Common Template ---
    def generate_impl(self, endpoint: str, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        last_detail = None

        if len(endpoint) < 1:
            raise RuntimeError("Missing endpoint")

        self.prevalidate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        for attempt in range(1, HF_MAX_ATTEMPTS + 1):
            headers = self.generate_header()
            payload = self.generate_payload(prompt,
                                            temperature=temperature,
                                            max_new_tokens=max_new_tokens)

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
                raise RuntimeError(f"HF error {r.status_code}: {body}")

            try:
                data = r.json()
            except Exception:
                raise RuntimeError(f"HF returned non-JSON: {(r.text or '')[:2000]}")

            return self.parse_results_text(data).strip()

        raise RuntimeError(f"HF model still loading (503). Last: {last_detail or 'n/a'}")

    # --- Custom hooks to be customized per strategy --

    # Get name of model adapter
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    # Calls to generate_impl while passing endpoint
    @abstractmethod
    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        raise NotImplementedError

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

    def name(self) -> str:
        return "huggingface_adapter"

    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        return self.generate_impl(self.endpoint_url, prompt, temperature=temperature, max_new_tokens=max_new_tokens)

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



class DeepInfraLlamaLLM(LLMStrategy):
    """
    Adapter for DeepInfra native inference endpoint for Llama instruct models:
      POST https://api.deepinfra.com/v1/inference/{model}
    Request example in prompt:
      { "input": "...", "stop": [...] }
    Response example in prompt:
      { "results": [ { "generated_text": "..." } ], ... }
    """

    def __init__(
            self,
            *,
            api_token: str,
            model: str = DEEPINFRA_DEFAULT_MODEL,
            base_url: str = DEEPINFRA_BASE_URL,
    ):
        self.api_token = api_token
        self.model = model
        self.endpoint_url = f"{base_url.rstrip('/')}/{model}"

        # Optional: allow callers to override stop list
        self.stop = LLAMA3_STOP

    def name(self) -> str:
        return "deepinfra_llama_adapter"

    # ---- core hook: route through base generate_impl() ----
    def generate(self, prompt: str, *, temperature: float, max_new_tokens: int) -> str:
        # uses your shared retry / 503 handling logic
        return self.generate_impl(
            self.endpoint_url,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    # ---- validation ----
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

    # ---- request building ----
    def generate_header(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _llama3_instruct_wrap(self, user_text: str) -> str:
        """
        DeepInfra expects the raw 'input' string. For Llama-3.x Instruct, your curl uses
        the tokenized chat wrapper. We auto-wrap unless the caller already provided it.
        """
        s = (user_text or "").strip()

        # Pass-through if user already supplied a fully wrapped conversation
        if "<|begin_of_text|>" in s or "<|start_header_id|>" in s:
            return s

        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{s}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate_payload(
            self,
            prompt: str,
            *,
            max_new_tokens: int = DEFAULT_MAX_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict[str, Any]:
        # Important: your base generate_impl currently calls generate_payload(prompt, temperature=...)
        # and does NOT pass max_new_tokens, so we must *not rely* on that param being provided.
        #
        # If you want max_new_tokens to take effect, update generate_impl to pass it through:
        #   payload = self.generate_payload(prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        #
        # Meanwhile: we keep defaults here; callers can still embed constraints in prompt if needed.
        return {
            "input": self._llama3_instruct_wrap(prompt),
            "stop": list(self.stop),
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
        }

    # ---- response parsing ----
    def parse_results_text(self, data) -> str:
        # Expected DeepInfra shape:
        # { "results": [ { "generated_text": "..." } ], ... }
        if isinstance(data, dict):
            results = data.get("results")
            if isinstance(results, list) and results:
                r0 = results[0] or {}
                txt = r0.get("generated_text")
                if isinstance(txt, str):
                    return txt

        # Defensive fallbacks (handy if DeepInfra changes shape or errors leak through)
        if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
            return data["generated_text"]
        if isinstance(data, str):
            return data
        return str(data)

    # ---- readiness / warmup ----
    async def is_model_ready(self, timeout=MODEL_TIMEOUT_SECS) -> bool:
        """
        Best-effort check: issue a tiny request with a short timeout.
        DeepInfra doesn't guarantee a separate health endpoint for inference models.
        """
        try:
            headers = self.generate_header()
            payload = {
                "input": self._llama3_instruct_wrap("ping"),
                "stop": list(self.stop),
                "temperature": 0.0,
                "max_new_tokens": 1,
            }
            r = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                _ = r.json()
                return True
            if r.status_code in (401, 403):
                return False
            # treat 503 / 5xx as "not ready"
            return False
        except Exception:
            return False

    async def send_warmup(self) -> bool:
        """
        Warm-up by forcing a short completion.
        """
        try:
            headers = self.generate_header()
            payload = {
                "input": self._llama3_instruct_wrap("Hello!"),
                "stop": list(self.stop),
                "temperature": 0.1,
                "max_new_tokens": 16,
            }
            r = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=60)
            return r.status_code == 200
        except Exception:
            return False

# -- LLM Factory builder --
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

        raise ValueError(f"Unknown LLM type: {kind}")


# -- Helpers --

def is_valid_model_type(type: str) -> bool:
    return isinstance(type, str) and type in MODEL_ADAPTOR_NAMES


