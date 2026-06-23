"""Pluggable LLM backend adapters for the Seeds of Truth RAG app.

This module defines a strategy-pattern abstraction over the various
large-language-model backends the app can talk to. :class:`LLMStrategy`
is the abstract base: it supplies a shared HTTP request loop
(:meth:`LLMStrategy.generate_impl`) with 503 cold-start retry handling,
and declares the per-backend hooks (header construction, payload
construction, response parsing, prevalidation, readiness/warmup checks).

Four concrete adapters implement that interface:

  - :class:`HFEndpointLLM`: a Hugging Face Inference Endpoint.
  - :class:`DeepInfraLlamaLLM`: DeepInfra's hosted Llama 3 models,
    using the Llama 3 instruct chat template (buffered).
  - :class:`DeepInfraStreamingLLM`: the same DeepInfra models via the
    OpenAI-compatible chat-completions API with ``stream=true``, consuming
    SSE deltas incrementally.
  - :class:`SparkCloudflareLLM`: a self-hosted Spark backend fronted by
    Cloudflare Access, using an OpenAI-style chat-completions API.
  - :class:`SimEndpointLLM`: a no-network simulation adapter for tests.

:class:`LLMFactory` builds the appropriate adapter from a short string
key (``"hf"``, ``"deepinfra"``, ``"spark"``, ``"sim"``), and
:func:`is_valid_model_type` validates such keys. Module-level constants
hold default generation parameters and per-backend configuration read
from environment variables.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

import logging_config
import model_prompts

# --- Shared params ---
DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.3

MODEL_TIMEOUT_SECS = 5

# DEPRECATED
# this is a server-end variable for selecting model prompt to use
# SYSTEM_PROMPT = model_prompts.DEEP_REPORTING_V1_SYSTEM_PROMPT
# SYSTEM_PROMPT = model_prompts.SMOKING_MAN_SYSTEM_PROMPT

model_logger = logging_config.get_logger("rag")

MODEL_ADAPTOR_NAMES = ["hf", "deepinfra", "deepinfra_stream", "spark", "sim", "vllm"]

# this is a server-end variable for selecting the hybrid rag algo
DEFAULT_HYBRID_RAG_ALGO = 1


# --- Hugging Face params ---
HF_ENDPOINT_URL = (
    "https://veecj6bnrlz86t6v.us-east-1.aws.endpoints.huggingface.cloud"
)
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

# --- DeepInfra (OpenAI-compatible streaming) params ---
# DeepInfraStreamingLLM talks to DeepInfra's OpenAI-compatible
# chat-completions endpoint with stream=true, so tokens arrive incrementally
# (keeping bytes flowing for any fronting proxy and defeating idle-read
# timeouts on long generations) instead of as one buffered blob.
#
# Why a second DeepInfra adapter rather than streaming the existing one:
# DeepInfraLlamaLLM posts to the *native* /v1/inference/{model} API and hand-
# wraps the Llama-3 instruct template. The streaming path instead uses the
# *OpenAI-compatible* /v1/openai/chat/completions API, where DeepInfra applies
# the model's chat template server-side, so we send plain chat messages and
# parse standard OpenAI SSE deltas (the same wire shape VLLMStreamingLLM
# consumes). The two coexist: the native one stays the buffered default,
# this one is the streaming variant.
#
# DeepInfra is a public managed API (no Cloudflare-Access wrapper), so unlike
# VLLMStreamingLLM this adapter sends no CF-Access headers and has no /health
# probe — readiness is a tiny one-token completion, matching DeepInfraLlamaLLM.
DEEPINFRA_OPENAI_BASE_URL = (
    os.getenv("DEEPINFRA_OPENAI_BASE_URL", "").strip()
    or "https://api.deepinfra.com/v1/openai"
)
# Read timeout = max gap allowed between two streamed chunks. DeepInfra emits
# tokens every <1s once decode begins; prefill on a long prompt can produce a
# longer dark window, so 60s is a generous ceiling.
DEEPINFRA_STREAM_READ_TIMEOUT_SECS = int(
    os.getenv("SOT_DEEPINFRA_READ_TIMEOUT_SECS", "60")
)
DEEPINFRA_STREAM_CONNECT_TIMEOUT_SECS = int(
    os.getenv("SOT_DEEPINFRA_CONNECT_TIMEOUT_SECS", "10")
)
# Hard cap on total wall-clock time per generate() call; defends against a
# pathological stream that never terminates.
DEEPINFRA_STREAM_TOTAL_TIMEOUT_SECS = int(
    os.getenv("SOT_DEEPINFRA_TOTAL_TIMEOUT_SECS", "600")
)

# --- Spark / Cloudflare wrapper params ---
SPARK_BASE_URL = "https://seedsoftruth.peerservice.org"
SPARK_SITE_API_KEY = os.getenv("SPARK_SITE_API_KEY", "").strip()
SPARK_CF_ACCESS_CLIENT_ID = os.getenv("SPARK_CF_ACCESS_CLIENT_ID", "").strip()
SPARK_CF_ACCESS_CLIENT_SECRET = os.getenv(
    "SPARK_CF_ACCESS_CLIENT_SECRET", ""
).strip()
SPARK_MODEL_NAME = os.getenv("SPARK_MODEL_NAME", "wtk_gamma_v9").strip()

# --- vLLM (OpenAI-compatible streaming) params ---
# Talks to a vLLM server's /v1/chat/completions endpoint with stream=true.
# The streaming response keeps bytes flowing across Cloudflare's edge so the
# 100-second Proxy Read Timeout never fires — see the design discussion in
# seedsoftruth_async_chat_design.md and the user-facing latency notes.
#
# Endpoint sharing with SparkCloudflareLLM:
# In our deployment the vLLM server IS the Spark backend — same DGX host,
# same Cloudflare Access posture, same /v1/chat/completions URL. The two
# adapters differ only in HOW they talk to it: SparkCloudflareLLM buffers
# the response (and gets cut off by Cloudflare's 100s read timeout on long
# generations), while VLLMStreamingLLM consumes SSE incrementally so the
# timeout can't fire.
#
# To make the migration zero-config, each SOT_VLLM_* env var falls back
# to its SPARK_* equivalent when unset. An unset SOT_VLLM_BASE_URL becomes
# the empty string, which is falsy, so SPARK_BASE_URL fills in. Set the
# SOT_VLLM_* explicitly only when vLLM lives on a different endpoint than
# your Spark wrapper points at.
def _vllm_env_chain(vllm_key: str, spark_key: str, hardcoded_default: str = "") -> str:
    """
    Pull the value for a VLLM_* config from the env, falling back to its
    SPARK_* equivalent and then to a final hardcoded default. We re-read
    the env vars (rather than relying on the module-level SPARK_* constants)
    because SPARK_BASE_URL is hardcoded at the top of this file rather than
    read from env at import — chaining straight through env avoids that
    inconsistency without us having to refactor the Spark setup itself.
    """
    return (
        os.getenv(vllm_key, "").strip()
        or os.getenv(spark_key, "").strip()
        or hardcoded_default
    )


# Env var naming: the SOT_ prefix matches the project's existing
# convention (cf. SOT_PASSWORD used by tools/prober.py). The Python
# constant names keep the shorter VLLM_* form since they're internal —
# only the env var STRINGS use SOT_VLLM_*.
VLLM_BASE_URL = _vllm_env_chain("SOT_VLLM_BASE_URL", "SPARK_BASE_URL", SPARK_BASE_URL)
VLLM_API_KEY = _vllm_env_chain("SOT_VLLM_API_KEY", "SPARK_SITE_API_KEY")
VLLM_MODEL_NAME = _vllm_env_chain("SOT_VLLM_MODEL_NAME", "SPARK_MODEL_NAME", "wtk_gamma_v9")
# Cloudflare Access headers — same defaults as Spark since they gate the
# same edge. Leave both env vars unset (no SPARK_* either) for an open
# or Tunnel'd vLLM endpoint.
VLLM_CF_ACCESS_CLIENT_ID = _vllm_env_chain(
    "SOT_VLLM_CF_ACCESS_CLIENT_ID", "SPARK_CF_ACCESS_CLIENT_ID"
)
VLLM_CF_ACCESS_CLIENT_SECRET = _vllm_env_chain(
    "SOT_VLLM_CF_ACCESS_CLIENT_SECRET", "SPARK_CF_ACCESS_CLIENT_SECRET"
)
# Read timeout = max gap allowed between two streamed chunks. vLLM normally
# emits tokens every <1s once decode begins; prefill on a long prompt can
# produce a longer dark window. 60s is a generous ceiling that still keeps
# us well under Cloudflare's 100s read timeout.
VLLM_READ_TIMEOUT_SECS = int(os.getenv("SOT_VLLM_READ_TIMEOUT_SECS", "60"))
VLLM_CONNECT_TIMEOUT_SECS = int(os.getenv("SOT_VLLM_CONNECT_TIMEOUT_SECS", "10"))
# Hard cap on total wall-clock time per generate() call, defends against a
# pathological stream that never terminates. 10 minutes matches the client
# polling ceiling so we don't keep a worker busy past what the UI will wait.
VLLM_TOTAL_TIMEOUT_SECS = int(os.getenv("SOT_VLLM_TOTAL_TIMEOUT_SECS", "600"))
# Health-check path. Default `/health` matches the Cloudflare-fronted
# wrapper this adapter typically sits behind (same shape as
# SparkCloudflareLLM.health_url). Override to `/v1/models` for a pure
# vLLM endpoint, or to any other path your reverse proxy exposes.
VLLM_HEALTH_PATH = os.getenv("SOT_VLLM_HEALTH_PATH", "/health").strip() or "/health"


def _iter_openai_sse_content(line_iter, *, total_timeout_secs=None, started_at=None):
    """Yield ``choices[0].delta.content`` strings from an OpenAI-style SSE
    line iterator.

    Shared by the streaming adapters' ``generate_stream`` generators. Mirrors
    the per-event parsing of their buffered ``generate`` loops, but yields
    each text fragment as it arrives instead of accumulating. Blank lines,
    non-``data:`` lines, comments, and malformed JSON are skipped; a
    ``data: [DONE]`` line ends iteration. When ``total_timeout_secs`` and
    ``started_at`` are supplied, iteration stops once the wall-clock budget is
    exceeded (defends against a stream that never terminates).

    Args:
        line_iter: Iterable of raw SSE text lines (e.g. ``r.iter_lines``).
        total_timeout_secs: Optional overall wall-clock budget in seconds.
        started_at: Optional ``time.time()`` value marking the request start.

    Yields:
        str: Non-empty content deltas, in order.
    """
    for raw in line_iter:
        if (
            total_timeout_secs is not None
            and started_at is not None
            and time.time() - started_at > total_timeout_secs
        ):
            break
        if not raw or not raw.startswith("data: "):
            continue
        payload_str = raw[len("data: "):]
        if payload_str == "[DONE]":
            break
        try:
            event = json.loads(payload_str)
        except ValueError:
            # Heartbeat / malformed line — skip rather than abort.
            continue
        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        c0 = choices[0] or {}
        delta = c0.get("delta") or {}
        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            yield piece


def get_system_prompt(prompt_type: int) -> str:
    """Return the system prompt for a given prompt-type index.

    Args:
        prompt_type: Index into ``model_prompts.MODEL_SYSTEM_PROMPTS``.

    Returns:
        str: The system prompt string at that index.

    Raises:
        RuntimeError: If ``prompt_type`` is out of range.
    """
    if prompt_type < 0 or prompt_type >= len(
        model_prompts.MODEL_SYSTEM_PROMPTS
    ):
        raise RuntimeError(f"Invalid prompt type: {prompt_type}")
    return model_prompts.MODEL_SYSTEM_PROMPTS[prompt_type]


class LLMStrategy(ABC):
    """Abstract base for an LLM backend adapter.

    Subclasses implement the per-backend hooks (header/payload
    construction, response parsing, prevalidation, readiness and
    warmup checks). The base class provides :meth:`generate_impl`,
    a shared HTTP POST loop with cold-start (HTTP 503) retry handling.
    """

    def generate_impl(
        self,
        endpoint: str,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Run a generation request with cold-start retry handling.

        Validates inputs via :meth:`prevalidate`, then POSTs the
        backend-specific payload to ``endpoint`` up to
        ``HF_MAX_ATTEMPTS`` times, sleeping and retrying on HTTP 503
        (model loading) responses.

        Args:
            endpoint: The fully qualified URL to POST to.
            prompt: The user (non-system) prompt to generate from.
            system_prompt: Optional system prompt passed through to the
                backend-specific payload builder.
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            str: The parsed, stripped generated text.

        Raises:
            RuntimeError: If the endpoint is empty, the backend
                returns a non-OK or non-JSON response, or the model
                is still loading after all retry attempts.
        """
        last_detail = None

        if len(endpoint) < 1:
            raise RuntimeError("Missing endpoint")

        self.prevalidate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )

        for attempt in range(1, HF_MAX_ATTEMPTS + 1):
            headers = self.generate_header()
            payload = self.generate_payload(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            r = requests.post(
                endpoint, headers=headers, json=payload, timeout=HF_TIMEOUT
            )

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
                raise RuntimeError(
                    f"LLM returned non-JSON: {(r.text or '')[:2000]}"
                )

            return self.parse_results_text(data).strip()

        raise RuntimeError(
            f"Model still loading (503). Last: {last_detail or 'n/a'}"
        )

    @abstractmethod
    def name(self) -> str:
        """Return a short human-readable name for this adapter.

        Returns:
            str: The adapter's identifying name.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Generate text from a prompt using this backend.

        ``prompt`` is the USER content only — the system prompt (if any)
        is passed separately via ``system_prompt``. Adapters that talk an
        OpenAI-shape API send it as a ``system`` role message; adapters
        that take a single string input prepend it. Callers must NOT
        bake the system prompt into ``prompt`` themselves (that produces
        duplicate system content on OpenAI-shape adapters — see the
        2026-05-29 bugfix in rag_controller.ask).

        Args:
            prompt: The user prompt to generate from.
            system_prompt: Optional system prompt sent separately.
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        raise NotImplementedError

    @abstractmethod
    def prevalidate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Validate request inputs before a generation call.

        Args:
            prompt: The user prompt to validate.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Raises:
            RuntimeError: If configuration or arguments are invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_header(self) -> Dict[str, str]:
        """Build the HTTP headers for a request to this backend.

        Returns:
            Dict[str, str]: The request headers (auth, content type,
            and any backend-specific headers).
        """
        raise NotImplementedError

    @abstractmethod
    def generate_payload(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Build the JSON request body for this backend.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt to include.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dict[str, Any]: The backend-specific request payload.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_results_text(self, data: Any) -> str:
        """Extract the generated text from a backend response.

        Args:
            data: The decoded JSON (or raw) response body.

        Returns:
            str: The generated text extracted from ``data``.
        """
        raise NotImplementedError

    @abstractmethod
    async def is_model_ready(self, timeout: int = MODEL_TIMEOUT_SECS) -> bool:
        """Check whether the backend is ready to serve requests.

        Args:
            timeout: Per-request timeout in seconds.

        Returns:
            bool: ``True`` if the model is ready, ``False`` otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_warmup(self) -> bool:
        """Send a small request to warm up a cold backend.

        Returns:
            bool: ``True`` if the warmup request succeeded, ``False``
            otherwise.
        """
        raise NotImplementedError


class HFEndpointLLM(LLMStrategy):
    """LLM adapter for a Hugging Face Inference Endpoint."""

    def __init__(self, endpoint_url: str, api_key: str):
        """Initialize the adapter.

        Args:
            endpoint_url: URL of the Hugging Face inference endpoint.
            api_key: Bearer token for the endpoint.
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def name(self) -> str:
        """Return the adapter name.

        Returns:
            str: ``"huggingface_adapter"``.
        """
        return "huggingface_adapter"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Generate text via the Hugging Face endpoint.

        Args:
            prompt: The user prompt to generate from.
            system_prompt: Optional system prompt sent separately.
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        return self.generate_impl(
            self.endpoint_url,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def prevalidate(
        self, prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        """Validate that the endpoint URL and API key are configured.

        Args:
            prompt: The user prompt (unused; present for interface
                compatibility).
            max_new_tokens: Maximum number of tokens to generate
                (unused).
            temperature: Sampling temperature (unused).

        Raises:
            RuntimeError: If the endpoint URL or API key is missing.
        """
        if not self.endpoint_url:
            raise RuntimeError("Missing HF_ENDPOINT_URL")
        if not self.api_key:
            raise RuntimeError("Missing HF_API_KEY")

    def generate_header(self) -> Dict[str, str]:
        """Build the request headers for the Hugging Face endpoint.

        Returns:
            Dict[str, str]: Headers with bearer auth and JSON content
            and accept types.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def generate_payload(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Build the JSON payload for the Hugging Face endpoint.

        HF Inference Endpoints take a single ``inputs`` string with no
        role separation, so the system prompt (if any) is prepended to
        the user content with a blank line.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt prepended to ``prompt``.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dict[str, Any]: A payload with ``inputs`` and a
            ``parameters`` block; ``return_full_text`` is ``False`` so
            only the completion is returned.
        """
        full_input = (
            f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        )
        return {
            "inputs": full_input,
            "parameters": {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "return_full_text": False,
            },
        }

    def parse_results_text(self, data: Any) -> str:
        """Extract generated text from a Hugging Face response.

        Handles the list-of-dicts and single-dict ``generated_text``
        shapes, an OpenAI-style ``choices`` shape, and a bare string,
        falling back to ``str(data)``.

        Args:
            data: The decoded response body.

        Returns:
            str: The extracted generated text.
        """
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if data[0].get("generated_text"):
                return str(data[0]["generated_text"])
        if isinstance(data, dict):
            if data.get("generated_text"):
                return str(data["generated_text"])
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                txt = (
                    c0.get("text")
                    or (c0.get("message") or {}).get("content")
                    or ""
                )
                if txt:
                    return str(txt)
        if isinstance(data, str):
            return data
        return str(data)

    async def is_model_ready(self, timeout: int = MODEL_TIMEOUT_SECS) -> bool:
        """Check the Hugging Face endpoint's health.

        POSTs a health-check payload and inspects the status code and
        body. A 503 is treated as a cold start (not ready).

        Args:
            timeout: Per-request timeout in seconds.

        Returns:
            bool: ``True`` only if the endpoint returns 200 with a
            ``health`` value of ``"ok"``; ``False`` for auth errors,
            cold starts, timeouts, or any other failure.
        """
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
                    model_logger.info(
                        "Model ready: Custom health response received"
                    )
                    return True
                else:
                    model_logger.info(
                        "Processed response but not explicit health OK"
                    )
                    return False
            elif r.status_code in (401, 403):
                model_logger.error(
                    f"Auth error {r.status_code}: Invalid HF_API_KEY?"
                )
                return False
            elif r.status_code == 503:
                model_logger.info(
                    "503: Model likely still loading (cold start)"
                )
                return False
            else:
                model_logger.warning(
                    f"Unexpected status: {r.status_code} - {r.text}"
                )
                return False
        except requests.Timeout:
            model_logger.warning("Health check timed out (model loading?)")
            return False
        except Exception as e:
            model_logger.warning(f"Health check failed: {e}")
            return False

    async def send_warmup(self) -> bool:
        """Send a small warmup request to the Hugging Face endpoint.

        Returns:
            bool: ``True`` if the warmup request returned HTTP 200,
            ``False`` otherwise (including on exceptions).
        """
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
    """LLM adapter for DeepInfra-hosted Llama 3 models."""

    def __init__(
        self,
        *,
        api_token: str,
        model: str = DEEPINFRA_DEFAULT_MODEL,
        base_url: str = DEEPINFRA_BASE_URL,
    ):
        """Initialize the adapter.

        Args:
            api_token: DeepInfra API bearer token.
            model: DeepInfra model identifier.
            base_url: Base URL of the DeepInfra inference API; the
                model id is appended to form the endpoint URL.
        """
        self.api_token = api_token
        self.model = model
        self.endpoint_url = f"{base_url.rstrip('/')}/{model}"
        self.stop = LLAMA3_STOP

    def name(self) -> str:
        """Return the adapter name.

        Returns:
            str: ``"deepinfra_llama_adapter"``.
        """
        return "deepinfra_llama_adapter"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Generate text via the DeepInfra endpoint.

        Args:
            prompt: The user prompt to generate from.
            system_prompt: Optional system prompt sent separately.
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        return self.generate_impl(
            self.endpoint_url,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def prevalidate(
        self, prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        """Validate configuration and request arguments.

        Checks that the endpoint URL, API token and model id are set,
        and that ``max_new_tokens``, ``temperature`` and ``prompt``
        have valid values.

        Args:
            prompt: The user prompt; must be a non-empty string.
            max_new_tokens: Must be an integer >= 1.
            temperature: Must be a float in the range [0, 2].

        Raises:
            RuntimeError: If any configuration value or argument is
                missing or invalid.
        """
        if not self.endpoint_url:
            raise RuntimeError("Missing DeepInfra endpoint_url")
        if not self.api_token:
            raise RuntimeError("Missing DEEPINFRA_TOKEN / api_token")
        if not self.model:
            raise RuntimeError("Missing DeepInfra model id")

        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise RuntimeError(
                f"max_new_tokens must be >= 1, got {max_new_tokens}"
            )

        t = float(temperature)
        if t < 0.0 or t > 2.0:
            raise RuntimeError(
                f"temperature must be in [0, 2], got {temperature}"
            )

        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("prompt must be a non-empty string")

    def generate_header(self) -> Dict[str, str]:
        """Build the request headers for the DeepInfra endpoint.

        Returns:
            Dict[str, str]: Headers with bearer auth and JSON content
            and accept types.
        """
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _llama3_instruct_wrap(
        self,
        user_text: str,
        *,
        system_text: Optional[str] = None,
    ) -> str:
        """Wrap a user turn (and optional system turn) in the Llama 3
        instruct chat template.

        If the text already contains Llama 3 special tokens it is
        returned unchanged — that path is used by callers that construct
        pre-formatted prompts themselves; otherwise it is wrapped with
        the begin-of-text, optional system, user and assistant header
        markers.

        Args:
            user_text: The raw user text to wrap.
            system_text: Optional system content to include as a system
                turn.

        Returns:
            str: The prompt formatted for a Llama 3 instruct model.
        """
        s = (user_text or "").strip()
        if "<|begin_of_text|>" in s or "<|start_header_id|>" in s:
            return s
        sys_block = ""
        if system_text:
            sys_block = (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_text}"
                "<|eot_id|>"
            )
        return (
            "<|begin_of_text|>"
            f"{sys_block}"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{s}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate_payload(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Build the JSON payload for the DeepInfra endpoint.

        The prompt is wrapped in the Llama 3 instruct template (with the
        optional system prompt as a system turn) and the Llama 3 stop
        sequences are included.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt included as a system turn.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dict[str, Any]: A payload with ``input``, ``stop``,
            ``temperature`` and ``max_new_tokens``.
        """
        return {
            "input": self._llama3_instruct_wrap(prompt, system_text=system_prompt),
            "stop": list(self.stop),
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
        }

    def parse_results_text(self, data: Any) -> str:
        """Extract generated text from a DeepInfra response.

        Handles the ``results`` list shape, a top-level
        ``generated_text``, an OpenAI-style ``choices`` shape, and a
        bare string, falling back to ``str(data)``.

        Args:
            data: The decoded response body.

        Returns:
            str: The extracted generated text.
        """
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
                txt = (
                    c0.get("text")
                    or (c0.get("message") or {}).get("content")
                    or ""
                )
                if txt:
                    return str(txt)
        if isinstance(data, str):
            return data
        return str(data)

    async def is_model_ready(self, timeout: int = MODEL_TIMEOUT_SECS) -> bool:
        """Check whether the DeepInfra model is ready.

        Sends a one-token ``ping`` request and treats an HTTP 200 as
        ready.

        Args:
            timeout: Per-request timeout in seconds.

        Returns:
            bool: ``True`` if the request returned HTTP 200, ``False``
            for auth errors, other statuses, or exceptions.
        """
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
        """Send a small warmup request to the DeepInfra endpoint.

        Returns:
            bool: ``True`` if the warmup request returned HTTP 200,
            ``False`` otherwise (including on exceptions).
        """
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
    """LLM adapter for a Spark backend behind Cloudflare Access.

    Talks to a self-hosted Spark deployment via an OpenAI-style
    chat-completions API, authenticating with both a site API key and
    Cloudflare Access service-token credentials.
    """

    def __init__(
        self,
        *,
        base_url: str,
        site_api_key: str,
        cf_access_client_id: str,
        cf_access_client_secret: str,
        model_name: str = SPARK_MODEL_NAME,
    ):
        """Initialize the adapter.

        Args:
            base_url: Base URL of the Spark deployment; trailing
                slashes are stripped. Used to derive the
                chat-completions and health URLs.
            site_api_key: Bearer token for the Spark site API.
            cf_access_client_id: Cloudflare Access client id.
            cf_access_client_secret: Cloudflare Access client secret.
            model_name: Name of the Spark model to request.
        """
        self.base_url = base_url.rstrip("/")
        self.site_api_key = site_api_key
        self.cf_access_client_id = cf_access_client_id
        self.cf_access_client_secret = cf_access_client_secret
        self.model_name = model_name
        self.endpoint_url = f"{self.base_url}/v1/chat/completions"
        self.health_url = f"{self.base_url}/health"

    def name(self) -> str:
        """Return the adapter name.

        Returns:
            str: ``"spark_cloudflare_adapter"``.
        """
        return "spark_cloudflare_adapter"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Generate text via the Spark chat-completions endpoint.

        Args:
            prompt: The user prompt to generate from.
            system_prompt: Optional system prompt sent separately.
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        return self.generate_impl(
            self.endpoint_url,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def prevalidate(
        self, prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        """Validate Spark configuration and request arguments.

        Checks that the base URL, site API key and Cloudflare Access
        credentials are set and that the prompt is a non-empty string.

        Args:
            prompt: The user prompt; must be a non-empty string.
            max_new_tokens: Maximum number of tokens to generate
                (unused).
            temperature: Sampling temperature (unused).

        Raises:
            RuntimeError: If any required configuration value is
                missing or the prompt is empty.
        """
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

    def generate_header(self) -> Dict[str, str]:
        """Build the request headers for the Spark endpoint.

        Returns:
            Dict[str, str]: Headers with site bearer auth, Cloudflare
            Access service-token headers, and JSON content and accept
            types.
        """
        return {
            "Authorization": f"Bearer {self.site_api_key}",
            "CF-Access-Client-Id": self.cf_access_client_id,
            "CF-Access-Client-Secret": self.cf_access_client_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def generate_payload(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Build the chat-completions payload for the Spark endpoint.

        Constructs an OpenAI-style ``messages`` array with an optional
        system message followed by the user prompt. The system content
        is supplied by the caller via ``system_prompt`` rather than baked
        into ``prompt`` (previously the system prompt appeared twice on
        the wire — once in the system role and again inside the user
        content; see the 2026-05-29 bugfix in rag_controller.ask).

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system message content.
            max_new_tokens: Maximum number of tokens to generate,
                passed as ``max_tokens``.
            temperature: Sampling temperature.

        Returns:
            Dict[str, Any]: A chat-completions request payload with
            ``model``, ``messages``, ``temperature`` and
            ``max_tokens``.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
        }

    def parse_results_text(self, data: Any) -> str:
        """Extract generated text from a Spark response.

        Handles the OpenAI-style ``choices`` shape, a top-level
        ``generated_text``, and a bare string, falling back to
        ``str(data)``.

        Args:
            data: The decoded response body.

        Returns:
            str: The extracted generated text.
        """
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                txt = (
                    c0.get("text")
                    or (c0.get("message") or {}).get("content")
                    or ""
                )
                if txt:
                    return str(txt)
            if isinstance(data.get("generated_text"), str):
                return data["generated_text"]
        if isinstance(data, str):
            return data
        return str(data)

    async def is_model_ready(self, timeout: int = MODEL_TIMEOUT_SECS) -> bool:
        """Check the Spark backend's health endpoint.

        Sends a GET to the health URL with Cloudflare Access headers.

        Args:
            timeout: Per-request timeout in seconds.

        Returns:
            bool: ``True`` if the health endpoint returned HTTP 200,
            ``False`` otherwise (including on exceptions).
        """
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
        """Send a small warmup chat request to the Spark endpoint.

        Returns:
            bool: ``True`` if the warmup request returned HTTP 200,
            ``False`` otherwise (including on exceptions).
        """
        try:
            r = requests.post(
                self.endpoint_url,
                headers=self.generate_header(),
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": model_prompts.MODEL_SYSTEM_PROMPTS[0],
                        },
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


class VLLMStreamingLLM(LLMStrategy):
    """
    Adapter for a vLLM server's OpenAI-compatible /v1/chat/completions
    endpoint, using `stream=true` SSE so Cloudflare's 100s Proxy Read
    Timeout never fires regardless of how long generation takes.

    How streaming defeats the 524:
      Cloudflare measures the gap between consecutive bytes received from
      the origin. With stream=true vLLM emits a Server-Sent Event roughly
      once per generated token (often several per second once decode is
      warm). Each event keeps Cloudflare's clock at zero, so total
      generation time becomes irrelevant — only the longest gap between
      tokens matters, and that's typically <1s.

    Wire format (each SSE event from vLLM):
        data: {"choices":[{"delta":{"content":"Hello"}}], ...}
        data: {"choices":[{"delta":{"content":" world"}}], ...}
        ...
        data: [DONE]

    The adapter accumulates the `delta.content` chunks and returns the
    concatenated string from generate(), so the rest of the pipeline
    (chat_with_corpus, worker, /api/job) is unchanged. We don't expose
    the chunks to the worker today — that would require an end-to-end
    streaming overhaul. The immediate goal is just to stop the 524.

    Why we don't reuse generate_impl(): that helper is the
    non-streaming /generate path used by HF/DeepInfra/Spark. Streaming
    needs `stream=True` on requests.post AND iter_lines() on the
    response object, which doesn't fit the buffer-and-parse contract.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str = VLLM_MODEL_NAME,
        cf_access_client_id: str = "",
        cf_access_client_secret: str = "",
        read_timeout_secs: int = VLLM_READ_TIMEOUT_SECS,
        connect_timeout_secs: int = VLLM_CONNECT_TIMEOUT_SECS,
        total_timeout_secs: int = VLLM_TOTAL_TIMEOUT_SECS,
        health_path: str = VLLM_HEALTH_PATH,
    ):
        """Initialize the streaming vLLM adapter.

        Derives the chat-completions endpoint, health-probe URL, and
        ``/v1/models`` URL from ``base_url``, and stores the auth and
        timeout settings used on every request.

        Args:
            base_url: Base URL of the vLLM (or Cloudflare-fronted) endpoint.
            api_key: Bearer token; sent as ``Authorization`` when non-empty.
            model_name: Model identifier passed in the request payload.
            cf_access_client_id: Cloudflare Access client id, if fronted.
            cf_access_client_secret: Cloudflare Access client secret.
            read_timeout_secs: Per-read socket timeout for streamed responses.
            connect_timeout_secs: Connection-establishment timeout.
            total_timeout_secs: Overall wall-clock budget for a request.
            health_path: Path to probe for readiness (default ``/health``).
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.cf_access_client_id = cf_access_client_id
        self.cf_access_client_secret = cf_access_client_secret
        self.read_timeout_secs = read_timeout_secs
        self.connect_timeout_secs = connect_timeout_secs
        self.total_timeout_secs = total_timeout_secs
        self.endpoint_url = f"{self.base_url}/v1/chat/completions"
        # Health check URL. The `health_path` parameter (driven by
        # SOT_VLLM_HEALTH_PATH at module load) determines which path we
        # probe. Default is /health, matching the Cloudflare-fronted
        # wrapper this adapter typically sits behind. Override to
        # /v1/models (or anything else) for pure vLLM deployments.
        primary = health_path if health_path.startswith("/") else f"/{health_path}"
        self.health_url = f"{self.base_url}{primary}"
        # Kept around as a convenience for anything that wants the raw
        # /v1/models URL specifically (e.g. diagnostic tooling).
        self.models_url = f"{self.base_url}/v1/models"

    def name(self) -> str:
        """Return the adapter's stable identifier string."""
        return "vllm_streaming_adapter"

    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ):
        """Stream a chat completion, yielding each text fragment as it arrives.

        The generator companion to :meth:`generate`: yields ``delta.content``
        fragments incrementally so a caller can forward them to the browser
        over Server-Sent Events. If an intermediary buffers the SSE into a
        single non-streaming body, the recovered text is yielded once.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system message; not defaulted here.
            temperature: Sampling temperature in [0, 2].
            max_new_tokens: Maximum tokens to generate (>= 1).

        Yields:
            str: Text fragments of the model's reply, in order.

        Raises:
            RuntimeError: On validation failure, connection error, or a
                non-OK HTTP status from the endpoint.
        """
        self.prevalidate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        headers = self.generate_header()
        payload = self.generate_payload(
            prompt, system_prompt=system_prompt,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        started_at = time.time()
        try:
            r = requests.post(
                self.endpoint_url, headers=headers, json=payload,
                timeout=(self.connect_timeout_secs, self.read_timeout_secs),
                stream=True,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"vLLM connection failed: {e}")

        if not r.ok:
            body = ""
            try:
                body = (r.text or "")[:2000]
            except Exception:
                pass
            r.close()
            raise RuntimeError(f"vLLM error {r.status_code}: {body}")

        content_type = (r.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in content_type:
            recovered = self._recover_buffered_text(r, content_type)
            if recovered:
                yield recovered
            return

        try:
            yield from _iter_openai_sse_content(
                r.iter_lines(decode_unicode=True),
                total_timeout_secs=self.total_timeout_secs,
                started_at=started_at,
            )
        finally:
            try:
                r.close()
            except Exception:
                pass

    def _recover_buffered_text(self, r, content_type: str) -> str:
        """Recover answer text from a non-SSE (buffered) streaming response.

        Used by :meth:`generate_stream` when a fronting wrapper returns a
        non-``text/event-stream`` body. Tries, in order: raw SSE mislabeled
        under the wrong Content-Type, the standard chat-completions JSON
        shape, then any string value carrying embedded SSE events (the
        error-envelope pathology this adapter was hardened against).

        Args:
            r: The already-fetched ``requests.Response`` (will be closed).
            content_type: The lower-cased response Content-Type.

        Returns:
            The recovered text, or "" if nothing usable was found.
        """
        body_text = ""
        try:
            body_text = r.text or ""
        except Exception:
            pass
        finally:
            try:
                r.close()
            except Exception:
                pass
        model_logger.warning(
            "vLLM streaming endpoint returned non-streaming Content-Type=%r; "
            "using buffered fallback.",
            content_type,
        )
        extracted = self._parse_sse_chunks(body_text).strip()
        if extracted:
            return extracted
        try:
            data = json.loads(body_text)
        except ValueError:
            return ""
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                text = self.parse_results_text(data).strip()
                if text:
                    return text
            for v in data.values():
                if isinstance(v, str) and "data: " in v:
                    extracted = self._parse_sse_chunks(v).strip()
                    if extracted:
                        return extracted
        return ""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Stream a chat completion and return the full concatenated text.

        Opens a streaming POST to the chat-completions endpoint, reads the
        SSE chunks as they arrive (keeping bytes flowing for any fronting
        proxy), and joins the deltas into the final answer.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system message; not defaulted here, to
                avoid duplicating a prompt the caller already supplied.
            temperature: Sampling temperature in [0, 2].
            max_new_tokens: Maximum tokens to generate (>= 1).

        Returns:
            The concatenated generated text.

        Raises:
            RuntimeError: On validation failure, connection error, or a
                non-OK HTTP status from the endpoint.
        """
        self.prevalidate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        headers = self.generate_header()
        payload = self.generate_payload(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        chunks: List[str] = []
        started_at = time.time()

        # stream=True on requests.post tells urllib3 NOT to download the
        # whole body before returning — the connection stays open and we
        # read it chunk-by-chunk via iter_lines(). This is what keeps
        # Cloudflare seeing bytes flow continuously.
        try:
            r = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=(self.connect_timeout_secs, self.read_timeout_secs),
                stream=True,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"vLLM connection failed: {e}")

        if not r.ok:
            body = ""
            try:
                body = (r.text or "")[:2000]
            except Exception:
                pass
            r.close()
            raise RuntimeError(f"vLLM error {r.status_code}: {body}")

        # ------------------------------------------------------------------
        # TEMPORARY MITIGATION (added 2026-05-29, expanded 2026-05-29) —
        # non-SSE response fallback.
        #
        # The upstream Cloudflare-fronted wrapper has been observed to NOT
        # forward vLLM's SSE chunks transparently. Two failure shapes seen:
        #
        #   1. Content-Type: application/json, body =
        #      {"choices":[{"message":{"content":"..."}}]} — the wrapper
        #      buffered the whole completion and returned the
        #      non-streaming OpenAI shape.
        #
        #   2. Content-Type: application/json, body =
        #      {"error":"data: {...}\n\ndata: {...}\n\n..."} — the wrapper
        #      DID receive vLLM's SSE chunks but stuffed them into an
        #      error envelope as a single string instead of streaming.
        #
        # When Content-Type is not text/event-stream, we try in order:
        #   A. Parse the raw body as if it were SSE (raw-SSE-under-wrong-
        #      Content-Type, in case the wrapper just mislabels).
        #   B. JSON-parse, then if `choices` exists use parse_results_text
        #      (case 1 above).
        #   C. JSON-parse, then if any string value contains `data: `
        #      events, parse those as SSE (case 2 above).
        # Fail loudly if none of those produce content.
        #
        # NOTE: this is a stopgap only. Long generations will still 524 at
        # Cloudflare because no bytes flow during the wait — the proper
        # fix is to make the upstream wrapper forward SSE chunks (or
        # remove it and point Cloudflare directly at vLLM).
        # ------------------------------------------------------------------
        content_type = (r.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in content_type:
            body_text = ""
            try:
                body_text = r.text or ""
            except Exception:
                pass
            finally:
                try:
                    r.close()
                except Exception:
                    pass
            model_logger.warning(
                "vLLM endpoint returned non-streaming Content-Type=%r; "
                "using buffered fallback. This bypasses the 524 mitigation "
                "— fix the upstream wrapper to forward SSE chunks.",
                content_type,
            )

            # Strategy A: body might already be raw SSE under the wrong
            # Content-Type. Try parsing directly first.
            extracted = self._parse_sse_chunks(body_text).strip()
            if extracted:
                return extracted

            # Parse JSON once for strategies B and C.
            try:
                data = json.loads(body_text)
            except ValueError:
                raise RuntimeError(
                    f"vLLM returned non-SSE Content-Type={content_type!r}, "
                    f"no SSE chunks found, and body was not valid JSON; "
                    f"first 500 chars: {body_text[:500]!r}"
                )

            if isinstance(data, dict):
                # Strategy B: standard OpenAI non-streaming completion shape.
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    text = self.parse_results_text(data).strip()
                    if text:
                        return text

                # Strategy C: error-envelope shape — scan all string values
                # for embedded SSE chunks. The "error" key has been observed
                # in practice; we tolerate any key name to stay resilient
                # to wrapper changes.
                for v in data.values():
                    if isinstance(v, str) and "data: " in v:
                        extracted = self._parse_sse_chunks(v).strip()
                        if extracted:
                            return extracted

            raise RuntimeError(
                f"vLLM returned non-SSE body with no recognizable content; "
                f"Content-Type={content_type!r}, first 500 chars: "
                f"{body_text[:500]!r}"
            )

        try:
            for raw in r.iter_lines(decode_unicode=True):
                # Total-wallclock guard — bounds runaway streams.
                if time.time() - started_at > self.total_timeout_secs:
                    model_logger.warning(
                        "vLLM stream exceeded total timeout %ss; closing.",
                        self.total_timeout_secs,
                    )
                    break

                # SSE: blank lines separate events; non-data lines (e.g.
                # event:, id:, comments) are ignored. vLLM only emits
                # `data: ...` events in practice.
                if not raw:
                    continue
                if not raw.startswith("data: "):
                    continue

                payload_str = raw[len("data: "):]
                # vLLM signals end-of-stream with `data: [DONE]`. Some
                # OpenAI-compat servers also emit a final usage chunk
                # before [DONE]; tolerate either order.
                if payload_str == "[DONE]":
                    break

                try:
                    event = json.loads(payload_str)
                except ValueError:
                    # Heartbeat / malformed line — skip rather than abort.
                    continue

                # Standard OpenAI chat-completions streaming shape:
                #   {"choices":[{"delta":{"content":"..."}, "finish_reason":null}]}
                choices = event.get("choices") if isinstance(event, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                c0 = choices[0] or {}
                delta = c0.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    chunks.append(piece)
                # finish_reason is non-null when the model stops; we
                # don't act on it specially because [DONE] follows.
        finally:
            try:
                r.close()
            except Exception:
                pass

        return "".join(chunks).strip()

    def _parse_sse_chunks(self, body: str) -> str:
        """Parse a complete SSE body (newline-separated `data: ...` events)
        into concatenated content.

        Used by the non-streaming fallback path in generate() when an
        upstream wrapper buffers SSE and re-emits it as a single string —
        either raw under the wrong Content-Type, or wrapped inside a JSON
        envelope. Mirrors the per-event logic of the main streaming loop
        but operates on an already-fetched string. Duplication is
        deliberately small for a fallback path; refactor only if a third
        caller appears.
        """
        pieces: List[str] = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            payload_str = line[len("data: "):].strip()
            if not payload_str or payload_str == "[DONE]":
                continue
            try:
                event = json.loads(payload_str)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            c0 = choices[0] or {}
            delta = c0.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                pieces.append(piece)
        return "".join(pieces)

    def prevalidate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        """Validate adapter config and request arguments before sending.

        Args:
            prompt: Must be a non-empty string.
            max_new_tokens: Must be an int >= 1.
            temperature: Must be a float in [0, 2].

        Raises:
            RuntimeError: If the base URL is unset or any argument is
                out of range.
        """
        if not self.base_url:
            raise RuntimeError("Missing SOT_VLLM_BASE_URL (or SPARK_BASE_URL fallback)")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("prompt must be a non-empty string")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise RuntimeError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        t = float(temperature)
        if t < 0.0 or t > 2.0:
            raise RuntimeError(f"temperature must be in [0, 2], got {temperature}")

    def generate_header(self) -> dict[str, str]:
        """Build request headers for a streaming chat call.

        Always sets JSON content type and an SSE ``Accept`` header; adds
        bearer auth and Cloudflare Access headers only when configured.

        Returns:
            The header dict to send with the request.
        """
        h = {
            "Content-Type": "application/json",
            # Explicit Accept header — some Cloudflare configs strip
            # streaming responses unless the client signals SSE intent.
            "Accept": "text/event-stream",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.cf_access_client_id:
            h["CF-Access-Client-Id"] = self.cf_access_client_id
        if self.cf_access_client_secret:
            h["CF-Access-Client-Secret"] = self.cf_access_client_secret
        return h

    def generate_payload(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict[str, Any]:
        """Build the OpenAI-shaped streaming chat-completions payload.

        Includes a system message only when ``system_prompt`` is provided
        (it is intentionally not defaulted, to avoid duplicating a prompt
        the caller already passed).

        Args:
            prompt: The user message content.
            system_prompt: Optional system message content.
            max_new_tokens: Maps to ``max_tokens`` in the payload.
            temperature: Sampling temperature.

        Returns:
            The request body dict with ``stream`` set to True.
        """
        # OpenAI-shape: system + user messages, plus stream:true. The
        # caller supplies the system content explicitly — we do NOT
        # default to MODEL_SYSTEM_PROMPTS[0] here. Doing so would
        # duplicate the system prompt when rag_controller has already
        # passed it explicitly. See the 2026-05-29 bugfix.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "stream": True,
        }

    def parse_results_text(self, data) -> str:
        """Extract answer text from a non-streaming response body.

        Provided to satisfy the adapter ABC; the streaming path does not
        use it. Reads ``choices[0].message.content`` when present and
        otherwise falls back to ``str(data)``.

        Args:
            data: A decoded response object (typically a dict).

        Returns:
            The extracted text, or an empty string when ``data`` is None.
        """
        # Not used by this adapter — streaming bypasses generate_impl /
        # parse_results_text. Implemented to satisfy the ABC and to give
        # a sensible fallback for any future caller that ever does a
        # non-streaming request.
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                msg = (c0.get("message") or {}).get("content")
                if isinstance(msg, str):
                    return msg
        return str(data) if data is not None else ""

    async def is_model_ready(self, timeout=MODEL_TIMEOUT_SECS) -> bool:
        """
        GET self.health_url and return True on 200, False otherwise.

        Header policy mirrors generate_header(): every auth/header
        identity we'd send on a real chat call also goes on the probe,
        so a 401 from a misconfigured bearer token surfaces here
        instead of waiting for the first chat to fail. Headers are sent
        only when configured — keeps the request small on open / tunneled
        endpoints.

        Logs the failure reason on the False path so a queue stuck on
        readiness doesn't require digging — the cause shows up directly
        in sot.log next to the worker's "still not ready" line.
        """
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.cf_access_client_id:
            headers["CF-Access-Client-Id"] = self.cf_access_client_id
        if self.cf_access_client_secret:
            headers["CF-Access-Client-Secret"] = self.cf_access_client_secret

        try:
            r = requests.get(self.health_url, headers=headers, timeout=timeout)
        except Exception as e:
            model_logger.info(
                "vLLM health probe at %s failed: %s", self.health_url, e,
            )
            return False

        if r.status_code != 200:
            model_logger.info(
                "vLLM health probe at %s returned %s",
                self.health_url, r.status_code,
            )
        return r.status_code == 200

    async def send_warmup(self) -> bool:
        """Issue a tiny streamed generation to warm up the endpoint.

        Acts as both a model warmup and a connectivity smoke test; the
        generated content is discarded.

        Returns:
            True if the warmup request returned HTTP 200, else False.
        """
        # One-token streamed generation acts as both warmup and a
        # connectivity smoke test. We don't care about the content.
        try:
            r = requests.post(
                self.endpoint_url,
                headers=self.generate_header(),
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": model_prompts.MODEL_SYSTEM_PROMPTS[0]},
                        {"role": "user", "content": HF_WARMUP_PROMPT},
                    ],
                    "temperature": 0.1,
                    "max_tokens": HF_WARMUP_MAX_NEW_TOKENS,
                    "stream": True,
                },
                timeout=(self.connect_timeout_secs, self.read_timeout_secs),
                stream=True,
            )
            ok = r.status_code == 200
            try:
                # Drain at least one chunk so the connection actually
                # exchanges bytes (some proxies upgrade a stream lazily).
                for _ in r.iter_lines(decode_unicode=True):
                    break
            finally:
                r.close()
            return ok
        except Exception as e:
            model_logger.warning(f"vLLM warm-up failed: {e}")
            return False


class DeepInfraStreamingLLM(LLMStrategy):
    """
    Streaming adapter for DeepInfra-hosted Llama 3 models via DeepInfra's
    OpenAI-compatible ``/v1/openai/chat/completions`` endpoint with
    ``stream=true``.

    Relationship to the other DeepInfra adapter:
        :class:`DeepInfraLlamaLLM` posts to the *native* ``/v1/inference``
        API and hand-wraps the Llama-3 instruct template, returning the whole
        completion in one buffered response. This adapter instead consumes the
        OpenAI-style SSE stream incrementally — DeepInfra applies the chat
        template server-side, so we send plain chat messages. Tokens arrive
        roughly once per generated token, keeping bytes flowing continuously
        (useful behind any proxy that enforces an idle-read timeout) and
        matching the wire shape :class:`VLLMStreamingLLM` already parses.

    Buffer-and-return contract:
        Like :class:`VLLMStreamingLLM`, this accumulates the ``delta.content``
        chunks and returns the concatenated string from :meth:`generate`, so
        the rest of the pipeline (chat_with_corpus, worker, /api/job) is
        unchanged. End-to-end streaming to the browser would be a separate
        change; the immediate goal is incremental, timeout-resistant
        generation against a reliable public endpoint.

    Wire format (each SSE event from DeepInfra):
        data: {"choices":[{"delta":{"content":"Hello"}}], ...}
        data: {"choices":[{"delta":{"content":" world"}}], ...}
        ...
        data: [DONE]
    """

    def __init__(
        self,
        *,
        api_token: str,
        model: str = DEEPINFRA_DEFAULT_MODEL,
        base_url: str = DEEPINFRA_OPENAI_BASE_URL,
        read_timeout_secs: int = DEEPINFRA_STREAM_READ_TIMEOUT_SECS,
        connect_timeout_secs: int = DEEPINFRA_STREAM_CONNECT_TIMEOUT_SECS,
        total_timeout_secs: int = DEEPINFRA_STREAM_TOTAL_TIMEOUT_SECS,
    ):
        """Initialize the streaming DeepInfra adapter.

        Args:
            api_token: DeepInfra API bearer token (``DEEPINFRA_TOKEN``).
            model: DeepInfra model identifier passed in the request payload.
            base_url: OpenAI-compatible API root; the chat-completions path is
                appended to form the endpoint URL.
            read_timeout_secs: Per-read socket timeout for streamed responses.
            connect_timeout_secs: Connection-establishment timeout.
            total_timeout_secs: Overall wall-clock budget for a request.
        """
        self.api_token = api_token
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.endpoint_url = f"{self.base_url}/chat/completions"
        self.read_timeout_secs = read_timeout_secs
        self.connect_timeout_secs = connect_timeout_secs
        self.total_timeout_secs = total_timeout_secs

    def name(self) -> str:
        """Return the adapter's stable identifier string.

        Returns:
            str: ``"deepinfra_streaming_adapter"``.
        """
        return "deepinfra_streaming_adapter"

    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ):
        """Stream a chat completion, yielding each text fragment as it arrives.

        The generator companion to :meth:`generate`: instead of accumulating
        the deltas and returning one string, it yields ``delta.content``
        fragments incrementally so a caller can forward them to the browser
        over Server-Sent Events. If an intermediary buffers the SSE into a
        single non-streaming JSON body, the recovered text is yielded once.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system message; not defaulted here.
            temperature: Sampling temperature in [0, 2].
            max_new_tokens: Maximum tokens to generate (>= 1).

        Yields:
            str: Text fragments of the model's reply, in order.

        Raises:
            RuntimeError: On validation failure, connection error, or a
                non-OK HTTP status from the endpoint.
        """
        self.prevalidate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        headers = self.generate_header()
        payload = self.generate_payload(
            prompt, system_prompt=system_prompt,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        started_at = time.time()
        try:
            r = requests.post(
                self.endpoint_url, headers=headers, json=payload,
                timeout=(self.connect_timeout_secs, self.read_timeout_secs),
                stream=True,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"DeepInfra connection failed: {e}")

        if not r.ok:
            body = ""
            try:
                body = (r.text or "")[:2000]
            except Exception:
                pass
            r.close()
            raise RuntimeError(f"DeepInfra error {r.status_code}: {body}")

        content_type = (r.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in content_type:
            # Buffered (non-streaming) fallback: recover the whole answer and
            # yield it once so a streaming caller still receives the text.
            recovered = self._recover_buffered_text(r, content_type)
            if recovered:
                yield recovered
            return

        try:
            yield from _iter_openai_sse_content(
                r.iter_lines(decode_unicode=True),
                total_timeout_secs=self.total_timeout_secs,
                started_at=started_at,
            )
        finally:
            try:
                r.close()
            except Exception:
                pass

    def _recover_buffered_text(self, r, content_type: str) -> str:
        """Recover answer text from a non-SSE (buffered) streaming response.

        Used by :meth:`generate_stream` when an intermediary returns a
        non-``text/event-stream`` body. Tries, in order: raw SSE mislabeled
        under the wrong Content-Type, the standard chat-completions JSON
        shape, then any string value carrying embedded SSE events.

        Args:
            r: The already-fetched ``requests.Response`` (will be closed).
            content_type: The lower-cased response Content-Type.

        Returns:
            The recovered text, or "" if nothing usable was found.
        """
        body_text = ""
        try:
            body_text = r.text or ""
        except Exception:
            pass
        finally:
            try:
                r.close()
            except Exception:
                pass
        model_logger.warning(
            "DeepInfra streaming endpoint returned non-streaming "
            "Content-Type=%r; using buffered fallback.",
            content_type,
        )
        extracted = self._parse_sse_chunks(body_text).strip()
        if extracted:
            return extracted
        try:
            data = json.loads(body_text)
        except ValueError:
            return ""
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                text = self.parse_results_text(data).strip()
                if text:
                    return text
            for v in data.values():
                if isinstance(v, str) and "data: " in v:
                    extracted = self._parse_sse_chunks(v).strip()
                    if extracted:
                        return extracted
        return ""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Stream a chat completion and return the full concatenated text.

        Opens a streaming POST to the chat-completions endpoint, reads the
        SSE chunks as they arrive, and joins the deltas into the final answer.

        Args:
            prompt: The user message to send.
            system_prompt: Optional system message; not defaulted here, to
                avoid duplicating a prompt the caller already supplied.
            temperature: Sampling temperature in [0, 2].
            max_new_tokens: Maximum tokens to generate (>= 1).

        Returns:
            The concatenated generated text.

        Raises:
            RuntimeError: On validation failure, connection error, or a
                non-OK HTTP status from the endpoint.
        """
        self.prevalidate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        headers = self.generate_header()
        payload = self.generate_payload(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        chunks: List[str] = []
        started_at = time.time()

        # stream=True tells urllib3 NOT to download the whole body before
        # returning — the connection stays open and we read it chunk-by-chunk
        # via iter_lines(), so bytes flow continuously.
        try:
            r = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=(self.connect_timeout_secs, self.read_timeout_secs),
                stream=True,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"DeepInfra connection failed: {e}")

        if not r.ok:
            body = ""
            try:
                body = (r.text or "")[:2000]
            except Exception:
                pass
            r.close()
            raise RuntimeError(f"DeepInfra error {r.status_code}: {body}")

        # Defensive non-SSE fallback: if some intermediary buffered the SSE and
        # re-emitted it as a single non-streaming JSON body, recover the text
        # rather than returning empty. Strategies mirror VLLMStreamingLLM:
        #   A. raw SSE under the wrong Content-Type, B. standard chat JSON,
        #   C. any string value carrying embedded SSE events.
        content_type = (r.headers.get("Content-Type") or "").lower()
        if "text/event-stream" not in content_type:
            body_text = ""
            try:
                body_text = r.text or ""
            except Exception:
                pass
            finally:
                try:
                    r.close()
                except Exception:
                    pass
            model_logger.warning(
                "DeepInfra streaming endpoint returned non-streaming "
                "Content-Type=%r; using buffered fallback.",
                content_type,
            )

            extracted = self._parse_sse_chunks(body_text).strip()
            if extracted:
                return extracted

            try:
                data = json.loads(body_text)
            except ValueError:
                raise RuntimeError(
                    f"DeepInfra returned non-SSE Content-Type={content_type!r}, "
                    f"no SSE chunks found, and body was not valid JSON; "
                    f"first 500 chars: {body_text[:500]!r}"
                )

            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    text = self.parse_results_text(data).strip()
                    if text:
                        return text
                for v in data.values():
                    if isinstance(v, str) and "data: " in v:
                        extracted = self._parse_sse_chunks(v).strip()
                        if extracted:
                            return extracted

            raise RuntimeError(
                f"DeepInfra returned non-SSE body with no recognizable content; "
                f"Content-Type={content_type!r}, first 500 chars: "
                f"{body_text[:500]!r}"
            )

        try:
            for raw in r.iter_lines(decode_unicode=True):
                # Total-wallclock guard — bounds runaway streams.
                if time.time() - started_at > self.total_timeout_secs:
                    model_logger.warning(
                        "DeepInfra stream exceeded total timeout %ss; closing.",
                        self.total_timeout_secs,
                    )
                    break

                if not raw:
                    continue
                if not raw.startswith("data: "):
                    continue

                payload_str = raw[len("data: "):]
                if payload_str == "[DONE]":
                    break

                try:
                    event = json.loads(payload_str)
                except ValueError:
                    # Heartbeat / malformed line — skip rather than abort.
                    continue

                choices = event.get("choices") if isinstance(event, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                c0 = choices[0] or {}
                delta = c0.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    chunks.append(piece)
        finally:
            try:
                r.close()
            except Exception:
                pass

        return "".join(chunks).strip()

    def _parse_sse_chunks(self, body: str) -> str:
        """Parse a complete SSE body (newline-separated ``data: ...`` events)
        into concatenated content.

        Used by the non-streaming fallback in :meth:`generate` when an
        upstream buffers SSE and re-emits it as a single string. Mirrors the
        per-event logic of the main streaming loop but operates on an
        already-fetched string.

        Args:
            body: The buffered SSE text.

        Returns:
            The concatenated ``delta.content`` values.
        """
        pieces: List[str] = []
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            payload_str = line[len("data: "):].strip()
            if not payload_str or payload_str == "[DONE]":
                continue
            try:
                event = json.loads(payload_str)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            c0 = choices[0] or {}
            delta = c0.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                pieces.append(piece)
        return "".join(pieces)

    def prevalidate(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
        """Validate adapter config and request arguments before sending.

        Args:
            prompt: Must be a non-empty string.
            max_new_tokens: Must be an int >= 1.
            temperature: Must be a float in [0, 2].

        Raises:
            RuntimeError: If the endpoint URL or API token is unset, or any
                argument is out of range.
        """
        if not self.endpoint_url:
            raise RuntimeError("Missing DeepInfra endpoint_url")
        if not self.api_token:
            raise RuntimeError("Missing DEEPINFRA_TOKEN / api_token")
        if not self.model_name:
            raise RuntimeError("Missing DeepInfra model id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("prompt must be a non-empty string")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise RuntimeError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        t = float(temperature)
        if t < 0.0 or t > 2.0:
            raise RuntimeError(f"temperature must be in [0, 2], got {temperature}")

    def generate_header(self) -> Dict[str, str]:
        """Build request headers for a streaming chat call.

        Sets JSON content type and an SSE ``Accept`` header, plus bearer auth
        from the DeepInfra token. No Cloudflare-Access headers — DeepInfra is
        a public managed API.

        Returns:
            Dict[str, str]: The header dict to send with the request.
        """
        h = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_token:
            h["Authorization"] = f"Bearer {self.api_token}"
        return h

    def generate_payload(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Build the OpenAI-shaped streaming chat-completions payload.

        Includes a system message only when ``system_prompt`` is provided (it
        is intentionally not defaulted, to avoid duplicating a prompt the
        caller already passed — see the 2026-05-29 bugfix in
        rag_controller.ask).

        Args:
            prompt: The user message content.
            system_prompt: Optional system message content.
            max_new_tokens: Maps to ``max_tokens`` in the payload.
            temperature: Sampling temperature.

        Returns:
            Dict[str, Any]: The request body with ``stream`` set to True.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "stream": True,
        }

    def parse_results_text(self, data: Any) -> str:
        """Extract answer text from a non-streaming response body.

        Provided to satisfy the adapter ABC and to back the buffered fallback
        in :meth:`generate`. Reads ``choices[0].message.content`` (or
        ``choices[0].text``) when present, else falls back to ``str(data)``.

        Args:
            data: A decoded response object (typically a dict).

        Returns:
            The extracted text, or an empty string when ``data`` is None.
        """
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                txt = (
                    (c0.get("message") or {}).get("content")
                    or c0.get("text")
                    or ""
                )
                if isinstance(txt, str) and txt:
                    return txt
        return str(data) if data is not None else ""

    async def is_model_ready(self, timeout: int = MODEL_TIMEOUT_SECS) -> bool:
        """Check whether the DeepInfra model is ready.

        Sends a tiny one-token, non-streaming chat completion and treats an
        HTTP 200 as ready. DeepInfra exposes no ``/health`` endpoint, so this
        mirrors :meth:`DeepInfraLlamaLLM.is_model_ready` (a one-token probe).

        Args:
            timeout: Per-request timeout in seconds.

        Returns:
            bool: ``True`` if the request returned HTTP 200, ``False`` for
            auth errors, other statuses, or exceptions.
        """
        try:
            r = requests.post(
                self.endpoint_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    **(
                        {"Authorization": f"Bearer {self.api_token}"}
                        if self.api_token
                        else {}
                    ),
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=timeout,
            )
            if r.status_code == 200:
                _ = r.json()
                return True
            if r.status_code in (401, 403):
                model_logger.info(
                    "DeepInfra readiness probe at %s returned %s (auth)",
                    self.endpoint_url, r.status_code,
                )
                return False
            model_logger.info(
                "DeepInfra readiness probe at %s returned %s",
                self.endpoint_url, r.status_code,
            )
            return False
        except Exception as e:
            model_logger.info(
                "DeepInfra readiness probe at %s failed: %s",
                self.endpoint_url, e,
            )
            return False

    async def send_warmup(self) -> bool:
        """Issue a tiny streamed generation to warm up the endpoint.

        Acts as both a model warmup and a connectivity smoke test; the
        generated content is discarded.

        Returns:
            True if the warmup request returned HTTP 200, else False.
        """
        try:
            r = requests.post(
                self.endpoint_url,
                headers=self.generate_header(),
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": HF_WARMUP_PROMPT}],
                    "temperature": 0.1,
                    "max_tokens": HF_WARMUP_MAX_NEW_TOKENS,
                    "stream": True,
                },
                timeout=(self.connect_timeout_secs, self.read_timeout_secs),
                stream=True,
            )
            ok = r.status_code == 200
            try:
                # Drain at least one chunk so the connection actually
                # exchanges bytes (some proxies upgrade a stream lazily).
                for _ in r.iter_lines(decode_unicode=True):
                    break
            finally:
                r.close()
            return ok
        except Exception as e:
            model_logger.warning(f"DeepInfra warm-up failed: {e}")
            return False


# This is simulated endpoint just for testing
class SimEndpointLLM(LLMStrategy):
    """No-network simulation LLM adapter for testing.

    Makes no HTTP requests: generation simply echoes the prompt and
    readiness/warmup always succeed.
    """

    def __init__(self):
        """Initialize the simulation adapter (no configuration)."""
        pass

    def name(self) -> str:
        """Return the adapter name.

        Returns:
            str: ``"sim_adapter"``.
        """
        return "sim_adapter"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Echo the prompt back as the generated text.

        If ``system_prompt`` is supplied, the echo is prefixed with a
        truncated marker so tests can confirm the system prompt arrived.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt (echoed as a prefix).
            temperature: Sampling temperature (ignored).
            max_new_tokens: Maximum number of tokens (ignored).

        Returns:
            str: The ``prompt`` unchanged (optionally prefixed).
        """
        if system_prompt:
            return f"[sys:{system_prompt[:40]}...]\n{prompt}"
        return prompt

    def prevalidate(
        self, prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        """No-op prevalidation for the simulation adapter.

        Args:
            prompt: The user prompt (ignored).
            max_new_tokens: Maximum number of tokens (ignored).
            temperature: Sampling temperature (ignored).

        Returns:
            str: An empty string.
        """
        return ""

    def generate_header(self) -> Dict[str, str]:
        """Return placeholder request headers.

        Returns:
            Dict[str, str]: A static set of headers; not used for any
            real request.
        """
        return {
            "Authorization": f"Bearer",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def generate_payload(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Dict[str, Any]:
        """Build a placeholder payload mirroring the HF format.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt (included in the payload).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dict[str, Any]: A payload with ``inputs``, ``system_prompt``
            and a ``parameters`` block; not used for any real request.
        """
        return {
            "inputs": prompt,
            "system_prompt": system_prompt,
            "parameters": {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "return_full_text": False,
            },
        }

    def parse_results_text(self, data: Any) -> str:
        """Return the response data coerced to a string.

        Args:
            data: The response body.

        Returns:
            str: ``str(data)``.
        """
        return str(data)

    async def is_model_ready(self, timeout: int = MODEL_TIMEOUT_SECS) -> bool:
        """Report the simulated model as always ready.

        Args:
            timeout: Per-request timeout in seconds (ignored).

        Returns:
            bool: Always ``True``.
        """
        return True

    async def send_warmup(self) -> bool:
        """Report the simulated warmup as always successful.

        Returns:
            bool: Always ``True``.
        """
        return True


class LLMFactory:
    """Factory for constructing :class:`LLMStrategy` adapters."""

    @staticmethod
    def create(kind: str) -> LLMStrategy:
        """Create an LLM adapter for the given backend key.

        Reads any required credentials and model identifiers from
        environment variables, falling back to module defaults.

        Args:
            kind: The backend key, one of ``"deepinfra"``, ``"hf"``,
                ``"spark"`` or ``"sim"``.

        Returns:
            LLMStrategy: A configured adapter instance.

        Raises:
            ValueError: If ``kind`` is not a recognized backend key.
        """
        if kind == "deepinfra":
            model_logger.info("Creating DeepInfra model adapter")
            return DeepInfraLlamaLLM(
                api_token=os.environ.get("DEEPINFRA_TOKEN", ""),
                model=os.environ.get(
                    "DEEPINFRA_MODEL", DEEPINFRA_DEFAULT_MODEL
                ),
            )

        if kind == "deepinfra_stream":
            model_logger.info(
                "Creating DeepInfra streaming model adapter (model=%s)",
                os.environ.get("DEEPINFRA_MODEL", DEEPINFRA_DEFAULT_MODEL),
            )
            return DeepInfraStreamingLLM(
                api_token=os.environ.get("DEEPINFRA_TOKEN", ""),
                model=os.environ.get(
                    "DEEPINFRA_MODEL", DEEPINFRA_DEFAULT_MODEL
                ),
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
                cf_access_client_id=os.environ.get(
                    "SPARK_CF_ACCESS_CLIENT_ID", ""
                ),
                cf_access_client_secret=os.environ.get(
                    "SPARK_CF_ACCESS_CLIENT_SECRET", ""
                ),
                model_name=os.environ.get("SPARK_MODEL_NAME", SPARK_MODEL_NAME),
            )

        if kind == "sim":
            model_logger.info("Creating Simulation model adapter")
            return SimEndpointLLM()

        if kind == "vllm":
            model_logger.info(
                "Creating vLLM streaming model adapter "
                "(base=%s, model=%s, cf_access=%s)",
                VLLM_BASE_URL or "<unset>",
                VLLM_MODEL_NAME or "<unset>",
                "yes" if VLLM_CF_ACCESS_CLIENT_ID else "no",
            )
            # All VLLM_* constants already incorporate the SPARK_* fallback
            # at module load (see the assignment block at the top of this
            # file). Pass them directly — re-reading env vars here would
            # bypass the fallback and is a footgun.
            return VLLMStreamingLLM(
                base_url=VLLM_BASE_URL,
                api_key=VLLM_API_KEY,
                model_name=VLLM_MODEL_NAME,
                cf_access_client_id=VLLM_CF_ACCESS_CLIENT_ID,
                cf_access_client_secret=VLLM_CF_ACCESS_CLIENT_SECRET,
            )

        raise ValueError(f"Unknown LLM type: {kind}")


def is_valid_model_type(type: str) -> bool:
    """Return whether a string is a recognized backend key.

    Args:
        type: The candidate backend key to check.

    Returns:
        bool: ``True`` if ``type`` is a string present in
        ``MODEL_ADAPTOR_NAMES``, ``False`` otherwise.
    """
    return isinstance(type, str) and type in MODEL_ADAPTOR_NAMES
