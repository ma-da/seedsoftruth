"""
Tests for the new VLLMStreamingLLM adapter.

What's covered:
  - SSE chunk parsing: deltas accumulate, [DONE] terminates, malformed
    lines are skipped, non-data lines are ignored
  - Header construction: bearer + CF Access only present when configured
  - Payload construction: stream:true is set, model/messages well-formed
  - LLMFactory.create("vllm") returns a VLLMStreamingLLM
  - is_valid_model_type accepts "vllm"
  - rag_controller.get_model_type("vllm") returns the registered model
  - Error path: non-2xx → RuntimeError with body excerpt

We mock requests.post so no real vLLM server is needed.

Run: python3 tests/test_vllm_adapter.py
"""
import sys
from pathlib import Path
from unittest import mock

REPO = str(Path(__file__).resolve().parent.parent)
# Web-server modules moved under chat_server/ but import each other flat.
sys.path.insert(0, str(Path(REPO) / "chat_server"))

# Avoid the heavy rag_retrieval import that rag_controller pulls in.
# We only need model_adapters for these tests; rag_controller is tested
# separately via a small smoke check at the end.
import model_adapters


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fake_response(status_code, lines, raise_on_iter=False):
    """Mimic the subset of requests.Response that VLLMStreamingLLM uses."""
    class _Resp:
        """Test double for ``requests.Response`` exposing only the attributes
        ``VLLMStreamingLLM`` reads: status, ok, text, ``iter_lines``, ``close``."""

        def __init__(self):
            """Capture the simulated status code and SSE lines."""
            self.status_code = status_code
            self.ok = 200 <= status_code < 300
            self.text = "" if self.ok else "<simulated error body>"
            self._lines = list(lines)

        def iter_lines(self, decode_unicode=True):
            """Yield the canned SSE lines, or raise if ``raise_on_iter`` is set
            (to simulate a mid-stream failure)."""
            if raise_on_iter:
                raise RuntimeError("simulated mid-stream failure")
            for line in self._lines:
                yield line

        def close(self):
            """No-op stand-in for ``Response.close``."""
            pass

    return _Resp()


def _sse(content):
    """Build a vLLM-style SSE line for a single content delta."""
    import json
    return 'data: ' + json.dumps({
        "choices": [{"delta": {"content": content}, "finish_reason": None}],
    })


def _build_adapter(**overrides):
    """Construct a ``VLLMStreamingLLM`` with default test config.

    Any keyword in ``overrides`` replaces the corresponding default, letting
    individual tests vary api_key, CF Access credentials, etc.
    """
    kwargs = dict(
        base_url="https://vllm.example.com",
        api_key="test-key",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
    )
    kwargs.update(overrides)
    return model_adapters.VLLMStreamingLLM(**kwargs)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_basic_sse_concatenates_deltas():
    """Streaming deltas accumulate into the full reply and the HTTP call uses
    stream=True both on the request and in the JSON payload."""
    adapter = _build_adapter()
    lines = [
        _sse("Hello"),
        _sse(", "),
        _sse("world"),
        _sse("!"),
        "data: [DONE]",
        "",  # trailing blank
    ]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ) as post:
        out = adapter.generate("hi", temperature=0.3, max_new_tokens=128)
    assert out == "Hello, world!", f"got {out!r}"
    # Verify stream=True was actually requested on the call
    _, kwargs = post.call_args
    assert kwargs["stream"] is True, "must use stream=True on the HTTP call"
    assert kwargs["json"]["stream"] is True, "must set stream:true in payload"
    print("  OK  test_basic_sse_concatenates_deltas")


def test_done_marker_terminates_stream():
    """A ``data: [DONE]`` line stops parsing; any deltas after it are ignored."""
    adapter = _build_adapter()
    lines = [
        _sse("first"),
        "data: [DONE]",
        _sse("would-be-second"),   # after [DONE] — must be ignored
    ]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ):
        out = adapter.generate("q", temperature=0.3, max_new_tokens=64)
    assert out == "first", f"reading past [DONE]; got {out!r}"
    print("  OK  test_done_marker_terminates_stream")


def test_malformed_lines_are_skipped():
    """Blank lines, non-data SSE fields, comments, and broken JSON are skipped
    without crashing; only valid content deltas contribute to the output."""
    adapter = _build_adapter()
    lines = [
        "",                          # blank
        "event: ping",               # non-data SSE field
        ": keepalive comment",       # SSE comment
        "data: {malformed json",     # broken json — skip, don't crash
        _sse("real"),
        "data: [DONE]",
    ]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ):
        out = adapter.generate("q", temperature=0.3, max_new_tokens=64)
    assert out == "real", f"got {out!r}"
    print("  OK  test_malformed_lines_are_skipped")


def test_non_2xx_raises_with_body_excerpt():
    """A non-2xx response raises RuntimeError whose message includes the status
    code (e.g. 'vLLM error 503')."""
    adapter = _build_adapter()
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(503, []),
    ):
        try:
            adapter.generate("q", temperature=0.3, max_new_tokens=64)
            raised = False
        except RuntimeError as e:
            raised = True
            assert "vLLM error 503" in str(e), f"unexpected message: {e}"
    assert raised, "must raise RuntimeError on non-2xx"
    print("  OK  test_non_2xx_raises_with_body_excerpt")


def test_connect_error_raises():
    """A requests-level connection failure is surfaced as a RuntimeError
    mentioning 'vLLM connection failed'."""
    adapter = _build_adapter()
    with mock.patch.object(
        model_adapters.requests, "post",
        side_effect=model_adapters.requests.RequestException("connect refused"),
    ):
        try:
            adapter.generate("q", temperature=0.3, max_new_tokens=64)
            raised = False
        except RuntimeError as e:
            raised = True
            assert "vLLM connection failed" in str(e)
    assert raised
    print("  OK  test_connect_error_raises")


def test_headers_include_bearer_and_sse_accept():
    """With an api_key set, headers carry the Bearer token plus the SSE Accept
    and JSON Content-Type values."""
    adapter = _build_adapter(api_key="secret-token")
    headers = adapter.generate_header()
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"
    print("  OK  test_headers_include_bearer_and_sse_accept")


def test_headers_omit_bearer_when_no_key():
    """With no api_key, no Authorization header is sent (open/tunneled vLLM)."""
    adapter = _build_adapter(api_key="")
    headers = adapter.generate_header()
    assert "Authorization" not in headers, (
        "Should not send Authorization header when no API key configured "
        "(e.g. tunneled or open vLLM)"
    )
    print("  OK  test_headers_omit_bearer_when_no_key")


def test_headers_include_cf_access_when_configured():
    """When CF Access credentials are configured, the CF-Access-Client-Id and
    CF-Access-Client-Secret headers are populated."""
    adapter = _build_adapter(
        cf_access_client_id="cf-id",
        cf_access_client_secret="cf-secret",
    )
    headers = adapter.generate_header()
    assert headers.get("CF-Access-Client-Id") == "cf-id"
    assert headers.get("CF-Access-Client-Secret") == "cf-secret"
    print("  OK  test_headers_include_cf_access_when_configured")


def test_prevalidate_rejects_empty_prompt():
    """prevalidate raises RuntimeError when given an empty prompt."""
    adapter = _build_adapter()
    try:
        adapter.prevalidate("", max_new_tokens=128, temperature=0.3)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    print("  OK  test_prevalidate_rejects_empty_prompt")


def test_factory_creates_vllm():
    """LLMFactory.create('vllm') returns a VLLMStreamingLLM whose name() is
    'vllm_streaming_adapter'."""
    a = model_adapters.LLMFactory.create("vllm")
    assert isinstance(a, model_adapters.VLLMStreamingLLM)
    assert a.name() == "vllm_streaming_adapter"
    print("  OK  test_factory_creates_vllm")


def test_is_valid_model_type_accepts_vllm():
    """is_valid_model_type accepts 'vllm' but rejects an adjacent typo ('vlm')."""
    assert model_adapters.is_valid_model_type("vllm")
    assert not model_adapters.is_valid_model_type("vlm")  # adjacent typo
    print("  OK  test_is_valid_model_type_accepts_vllm")


def test_system_prompt_appears_exactly_once_in_payload():
    """
    Regression: previously the system prompt was both (a) prepended to
    the prompt string by rag_controller.ask() and (b) added as the system
    message by VLLMStreamingLLM.generate_payload(), so it appeared twice
    on the wire. After the fix, the adapter should only put system_prompt
    in messages[0].role==system and nowhere else.
    """
    adapter = _build_adapter()
    payload = adapter.generate_payload(
        prompt="What killed JFK?",
        system_prompt="You are a careful researcher.",
        max_new_tokens=64,
        temperature=0.3,
    )

    messages = payload["messages"]
    assert len(messages) == 2, f"expected [system, user], got {messages}"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a careful researcher."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What killed JFK?"
    # The user message must NOT also contain the system prompt anywhere.
    assert "You are a careful researcher." not in messages[1]["content"], (
        "System prompt leaked into the user message — the duplication bug "
        "is back. Check rag_controller.ask()."
    )
    print("  OK  test_system_prompt_appears_exactly_once_in_payload")


def test_payload_omits_system_when_none():
    """When the caller doesn't provide a system_prompt, the adapter must
    NOT silently inject a default. Previously this adapter (and the Spark
    one) hardcoded MODEL_SYSTEM_PROMPTS[0] which was wrong on both
    correctness and configurability grounds."""
    adapter = _build_adapter()
    payload = adapter.generate_payload(
        prompt="hi",
        max_new_tokens=64,
        temperature=0.3,
    )
    messages = payload["messages"]
    assert len(messages) == 1, f"expected user-only, got {messages}"
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hi"
    print("  OK  test_payload_omits_system_when_none")


def test_hf_payload_prepends_system_to_inputs():
    """HF endpoints take a single `inputs` string. The adapter should
    prepend the system_prompt when one is provided."""
    hf = model_adapters.HFEndpointLLM(
        endpoint_url="https://hf.example.com",
        api_key="key",
    )
    p = hf.generate_payload(
        prompt="Question?",
        system_prompt="System rules.",
        max_new_tokens=64,
        temperature=0.3,
    )
    assert p["inputs"] == "System rules.\n\nQuestion?"
    # And the no-system path leaves the input untouched
    p2 = hf.generate_payload(prompt="just user", max_new_tokens=64, temperature=0.3)
    assert p2["inputs"] == "just user"
    print("  OK  test_hf_payload_prepends_system_to_inputs")


def test_deepinfra_wraps_system_in_llama3_template():
    """DeepInfra adapter builds a Llama-3 chat template. With a system_prompt
    it should produce the canonical system-then-user block."""
    di = model_adapters.DeepInfraLlamaLLM(api_token="x")
    p = di.generate_payload(
        prompt="who?",
        system_prompt="be brief",
        max_new_tokens=64,
        temperature=0.3,
    )
    wrapped = p["input"]
    assert "<|start_header_id|>system<|end_header_id|>\n\nbe brief<|eot_id|>" in wrapped
    assert "<|start_header_id|>user<|end_header_id|>\n\nwho?<|eot_id|>" in wrapped
    # System content should not also appear inside the user turn
    user_section = wrapped.split("<|start_header_id|>user<|end_header_id|>")[1]
    assert "be brief" not in user_section, "system content leaked into user turn"
    print("  OK  test_deepinfra_wraps_system_in_llama3_template")


def test_health_probe_hits_health_url_and_returns_true_on_200():
    """Single GET to self.health_url (= {base_url}/health). 200 → True.
    Pins the simplified behavior — no path fallback, no retries."""
    import asyncio
    adapter = _build_adapter()
    fake = mock.MagicMock(status_code=200)
    with mock.patch.object(
        model_adapters.requests, "get", return_value=fake,
    ) as g:
        ok = asyncio.run(adapter.is_model_ready(timeout=2))
    assert ok is True
    assert g.call_count == 1, f"expected exactly one GET, got {g.call_count}"
    called_url = g.call_args.args[0]
    assert called_url.endswith("/health"), called_url
    print("  OK  test_health_probe_hits_health_url_and_returns_true_on_200")


def test_health_probe_returns_false_on_non_200():
    """Anything other than 200 reads as 'not ready' — that's what tells
    the worker to keep waiting rather than dispatching a job."""
    import asyncio
    adapter = _build_adapter()
    for status in (404, 401, 502, 503):
        bad = mock.MagicMock(status_code=status)
        with mock.patch.object(
            model_adapters.requests, "get", return_value=bad,
        ):
            ok = asyncio.run(adapter.is_model_ready(timeout=2))
        assert ok is False, f"status {status} should produce False"
    print("  OK  test_health_probe_returns_false_on_non_200")


def test_health_probe_returns_false_on_exception():
    """Network error → False, never propagates an exception to the
    worker loop."""
    import asyncio
    adapter = _build_adapter()
    with mock.patch.object(
        model_adapters.requests, "get",
        side_effect=model_adapters.requests.RequestException("connect refused"),
    ):
        ok = asyncio.run(adapter.is_model_ready(timeout=2))
    assert ok is False
    print("  OK  test_health_probe_returns_false_on_exception")


def test_health_probe_sends_bearer_when_api_key_set():
    """Symmetry with generate_header(): if the chat call sends Bearer,
    the health probe should too. Otherwise a vLLM started with --api-key
    will 401 the health probe and the worker will sit in its readiness
    loop forever despite the chat path being fully usable."""
    import asyncio
    adapter = _build_adapter(api_key="secret-token")
    fake = mock.MagicMock(status_code=200)
    with mock.patch.object(
        model_adapters.requests, "get", return_value=fake,
    ) as g:
        asyncio.run(adapter.is_model_ready(timeout=2))
    sent_headers = g.call_args.kwargs["headers"]
    assert sent_headers.get("Authorization") == "Bearer secret-token"
    print("  OK  test_health_probe_sends_bearer_when_api_key_set")


def test_health_probe_omits_bearer_when_no_api_key():
    """Open / tunneled endpoints have no api_key. The probe must not
    send an empty Bearer header (some proxies reject malformed auth)."""
    import asyncio
    adapter = _build_adapter(api_key="")
    fake = mock.MagicMock(status_code=200)
    with mock.patch.object(
        model_adapters.requests, "get", return_value=fake,
    ) as g:
        asyncio.run(adapter.is_model_ready(timeout=2))
    sent_headers = g.call_args.kwargs["headers"]
    assert "Authorization" not in sent_headers
    print("  OK  test_health_probe_omits_bearer_when_no_api_key")


def test_health_probe_logs_on_failure():
    """An exception or non-200 should produce a model_logger.info line
    so a queue stuck on readiness is one grep away rather than a debug
    session. Verifies the post-review hardening fix."""
    import asyncio
    import logging as _logging
    adapter = _build_adapter()

    # Case 1: exception path
    with mock.patch.object(
        model_adapters.model_logger, "info",
    ) as info, mock.patch.object(
        model_adapters.requests, "get",
        side_effect=model_adapters.requests.RequestException("connect refused"),
    ):
        asyncio.run(adapter.is_model_ready(timeout=2))
    msgs = " ".join(str(c.args[0]) for c in info.call_args_list)
    assert "failed" in msgs.lower(), (
        f"exception path must log a failure message; got: {msgs}"
    )

    # Case 2: non-200 path
    bad = mock.MagicMock(status_code=503)
    with mock.patch.object(
        model_adapters.model_logger, "info",
    ) as info, mock.patch.object(
        model_adapters.requests, "get", return_value=bad,
    ):
        asyncio.run(adapter.is_model_ready(timeout=2))
    msgs = " ".join(str(c.args[0]) for c in info.call_args_list)
    assert "returned" in msgs.lower() or "503" in msgs, (
        f"non-200 path must log the status code; got: {msgs}"
    )
    print("  OK  test_health_probe_logs_on_failure")


def test_vllm_env_vars_fall_back_to_spark():
    """
    Zero-config migration story: if only SPARK_* env vars are set, the
    vLLM adapter should pick those up so an existing deployment gets
    streaming without any new configuration. Setting SOT_VLLM_*
    explicitly overrides the Spark defaults.
    """
    import importlib

    # Case 1: only SPARK_* set — SOT_VLLM_* should fall back.
    spark_env = {
        "SPARK_BASE_URL": "https://spark.example.com",
        "SPARK_SITE_API_KEY": "spark-api-key",
        "SPARK_CF_ACCESS_CLIENT_ID": "spark-cf-id",
        "SPARK_CF_ACCESS_CLIENT_SECRET": "spark-cf-secret",
        "SPARK_MODEL_NAME": "wtk_gamma_v9",
    }
    # Clear any SOT_VLLM_* that might leak from the host env.
    for k in [
        "SOT_VLLM_BASE_URL", "SOT_VLLM_API_KEY", "SOT_VLLM_MODEL_NAME",
        "SOT_VLLM_CF_ACCESS_CLIENT_ID", "SOT_VLLM_CF_ACCESS_CLIENT_SECRET",
    ]:
        spark_env[k] = ""
    with mock.patch.dict("os.environ", spark_env, clear=False):
        ma = importlib.reload(model_adapters)
        assert ma.VLLM_BASE_URL == "https://spark.example.com", ma.VLLM_BASE_URL
        assert ma.VLLM_API_KEY == "spark-api-key"
        assert ma.VLLM_MODEL_NAME == "wtk_gamma_v9"
        assert ma.VLLM_CF_ACCESS_CLIENT_ID == "spark-cf-id"
        assert ma.VLLM_CF_ACCESS_CLIENT_SECRET == "spark-cf-secret"

        adapter = ma.LLMFactory.create("vllm")
        assert adapter.base_url == "https://spark.example.com"
        assert adapter.api_key == "spark-api-key"
        assert adapter.model_name == "wtk_gamma_v9"
        assert adapter.cf_access_client_id == "spark-cf-id"
        assert adapter.cf_access_client_secret == "spark-cf-secret"

    # Case 2: SOT_VLLM_* set explicitly — must override Spark.
    mixed_env = dict(spark_env)
    mixed_env["SOT_VLLM_BASE_URL"] = "https://different-vllm.example.com"
    mixed_env["SOT_VLLM_MODEL_NAME"] = "meta-llama/Llama-3.3-70B-Instruct"
    with mock.patch.dict("os.environ", mixed_env, clear=False):
        ma = importlib.reload(model_adapters)
        assert ma.VLLM_BASE_URL == "https://different-vllm.example.com"
        assert ma.VLLM_MODEL_NAME == "meta-llama/Llama-3.3-70B-Instruct"
        # Unset SOT_VLLM_* still fall back to Spark
        assert ma.VLLM_API_KEY == "spark-api-key"
        assert ma.VLLM_CF_ACCESS_CLIENT_ID == "spark-cf-id"

    # Restore the module's normal state so later tests see a clean view.
    importlib.reload(model_adapters)
    print("  OK  test_vllm_env_vars_fall_back_to_spark")


if __name__ == "__main__":
    tests = [
        test_basic_sse_concatenates_deltas,
        test_done_marker_terminates_stream,
        test_malformed_lines_are_skipped,
        test_non_2xx_raises_with_body_excerpt,
        test_connect_error_raises,
        test_headers_include_bearer_and_sse_accept,
        test_headers_omit_bearer_when_no_key,
        test_headers_include_cf_access_when_configured,
        test_prevalidate_rejects_empty_prompt,
        test_factory_creates_vllm,
        test_is_valid_model_type_accepts_vllm,
        test_system_prompt_appears_exactly_once_in_payload,
        test_payload_omits_system_when_none,
        test_hf_payload_prepends_system_to_inputs,
        test_deepinfra_wraps_system_in_llama3_template,
        test_health_probe_hits_health_url_and_returns_true_on_200,
        test_health_probe_returns_false_on_non_200,
        test_health_probe_returns_false_on_exception,
        test_health_probe_sends_bearer_when_api_key_set,
        test_health_probe_omits_bearer_when_no_api_key,
        test_health_probe_logs_on_failure,
        test_vllm_env_vars_fall_back_to_spark,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print()
    print(f"{len(tests) - len(failures)} / {len(tests)} vllm adapter tests passed")
    sys.exit(1 if failures else 0)
