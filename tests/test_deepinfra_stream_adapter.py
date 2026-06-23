"""
Tests for the DeepInfraStreamingLLM adapter.

What's covered:
  - SSE chunk parsing: deltas accumulate, [DONE] terminates, malformed
    lines are skipped, non-data lines are ignored
  - stream=True is set both on the HTTP call and in the JSON payload
  - Header construction: bearer present when token set, SSE Accept set,
    and NO Cloudflare-Access headers (DeepInfra is a public API)
  - Payload construction: system message appears once / omitted when None
  - Non-SSE buffered fallback (Content-Type: application/json)
  - Error path: non-2xx -> RuntimeError with 'DeepInfra error <code>'
  - Connection error -> RuntimeError with 'DeepInfra connection failed'
  - Readiness probe: POST one-token completion, 200 -> True, else False
  - LLMFactory.create('deepinfra_stream') returns a DeepInfraStreamingLLM
  - is_valid_model_type accepts 'deepinfra_stream'
  - rag_controller.get_model_type('deepinfra_stream') resolves the adapter

We mock requests.post so no real DeepInfra endpoint is needed.

Run: python3 tests/test_deepinfra_stream_adapter.py
"""
import json
import sys
from pathlib import Path
from unittest import mock

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(REPO) / "chat_server"))

import model_adapters


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fake_response(status_code, lines, *, content_type="text/event-stream",
                   body_text=None, raise_on_iter=False):
    """Mimic the subset of requests.Response that DeepInfraStreamingLLM uses."""
    class _Resp:
        def __init__(self):
            self.status_code = status_code
            self.ok = 200 <= status_code < 300
            self.headers = {"Content-Type": content_type}
            if body_text is not None:
                self.text = body_text
            else:
                self.text = "" if self.ok else "<simulated error body>"
            self._lines = list(lines)

        def iter_lines(self, decode_unicode=True):
            if raise_on_iter:
                raise RuntimeError("simulated mid-stream failure")
            for line in self._lines:
                yield line

        def json(self):
            return json.loads(self.text) if self.text else {}

        def close(self):
            pass

    return _Resp()


def _sse(content):
    """Build an OpenAI-style SSE line for a single content delta."""
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
    )


def _build_adapter(**overrides):
    kwargs = dict(
        api_token="test-token",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
    kwargs.update(overrides)
    return model_adapters.DeepInfraStreamingLLM(**kwargs)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_basic_sse_concatenates_deltas():
    adapter = _build_adapter()
    lines = [_sse("Hello"), _sse(", "), _sse("world"), _sse("!"),
             "data: [DONE]", ""]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ) as post:
        out = adapter.generate("hi", temperature=0.3, max_new_tokens=128)
    assert out == "Hello, world!", f"got {out!r}"
    _, kwargs = post.call_args
    assert kwargs["stream"] is True, "must use stream=True on the HTTP call"
    assert kwargs["json"]["stream"] is True, "must set stream:true in payload"
    # Targets the OpenAI-compatible chat-completions endpoint.
    called_url = post.call_args.args[0]
    assert called_url.endswith("/v1/openai/chat/completions"), called_url
    print("  OK  test_basic_sse_concatenates_deltas")


def test_done_marker_terminates_stream():
    adapter = _build_adapter()
    lines = [_sse("first"), "data: [DONE]", _sse("would-be-second")]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ):
        out = adapter.generate("q", temperature=0.3, max_new_tokens=64)
    assert out == "first", f"reading past [DONE]; got {out!r}"
    print("  OK  test_done_marker_terminates_stream")


def test_malformed_lines_are_skipped():
    adapter = _build_adapter()
    lines = [
        "", "event: ping", ": keepalive comment",
        "data: {malformed json", _sse("real"), "data: [DONE]",
    ]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ):
        out = adapter.generate("q", temperature=0.3, max_new_tokens=64)
    assert out == "real", f"got {out!r}"
    print("  OK  test_malformed_lines_are_skipped")


def test_non_sse_buffered_fallback():
    """A buffered application/json chat-completions body still yields text."""
    adapter = _build_adapter()
    body = json.dumps({"choices": [{"message": {"content": "buffered answer"}}]})
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, [], content_type="application/json",
                                    body_text=body),
    ):
        out = adapter.generate("q", temperature=0.3, max_new_tokens=64)
    assert out == "buffered answer", f"got {out!r}"
    print("  OK  test_non_sse_buffered_fallback")


def test_non_2xx_raises_with_body_excerpt():
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
            assert "DeepInfra error 503" in str(e), f"unexpected message: {e}"
    assert raised, "must raise RuntimeError on non-2xx"
    print("  OK  test_non_2xx_raises_with_body_excerpt")


def test_connect_error_raises():
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
            assert "DeepInfra connection failed" in str(e)
    assert raised
    print("  OK  test_connect_error_raises")


def test_headers_include_bearer_and_sse_accept_no_cf():
    adapter = _build_adapter(api_token="secret-token")
    headers = adapter.generate_header()
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"
    # DeepInfra is a public API — no Cloudflare-Access headers.
    assert "CF-Access-Client-Id" not in headers
    assert "CF-Access-Client-Secret" not in headers
    print("  OK  test_headers_include_bearer_and_sse_accept_no_cf")


def test_prevalidate_rejects_empty_prompt_and_missing_token():
    adapter = _build_adapter()
    for bad in [
        lambda: adapter.prevalidate("", max_new_tokens=64, temperature=0.3),
        lambda: adapter.prevalidate("q", max_new_tokens=0, temperature=0.3),
        lambda: adapter.prevalidate("q", max_new_tokens=64, temperature=9),
    ]:
        try:
            bad()
            raised = False
        except RuntimeError:
            raised = True
        assert raised
    # Missing token
    no_tok = _build_adapter(api_token="")
    try:
        no_tok.prevalidate("q", max_new_tokens=64, temperature=0.3)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "empty api_token must fail prevalidation"
    print("  OK  test_prevalidate_rejects_empty_prompt_and_missing_token")


def test_payload_system_appears_once_and_stream_true():
    adapter = _build_adapter()
    p = adapter.generate_payload(
        prompt="What killed JFK?",
        system_prompt="You are a careful researcher.",
        max_new_tokens=64, temperature=0.3,
    )
    assert p["stream"] is True
    messages = p["messages"]
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "You are a careful researcher."}
    assert messages[1] == {"role": "user", "content": "What killed JFK?"}
    assert "You are a careful researcher." not in messages[1]["content"]
    print("  OK  test_payload_system_appears_once_and_stream_true")


def test_payload_omits_system_when_none():
    adapter = _build_adapter()
    p = adapter.generate_payload(prompt="hi", max_new_tokens=64, temperature=0.3)
    messages = p["messages"]
    assert len(messages) == 1 and messages[0] == {"role": "user", "content": "hi"}
    print("  OK  test_payload_omits_system_when_none")


def test_factory_creates_deepinfra_stream():
    import os
    with mock.patch.dict("os.environ", {"DEEPINFRA_TOKEN": "x"}, clear=False):
        a = model_adapters.LLMFactory.create("deepinfra_stream")
    assert isinstance(a, model_adapters.DeepInfraStreamingLLM)
    assert a.name() == "deepinfra_streaming_adapter"
    assert a.endpoint_url.endswith("/v1/openai/chat/completions")
    print("  OK  test_factory_creates_deepinfra_stream")


def test_is_valid_model_type_accepts_deepinfra_stream():
    assert model_adapters.is_valid_model_type("deepinfra_stream")
    assert not model_adapters.is_valid_model_type("deepinfra_streaming")
    print("  OK  test_is_valid_model_type_accepts_deepinfra_stream")


def test_readiness_probe_true_on_200_false_otherwise():
    import asyncio
    adapter = _build_adapter()

    ok_resp = _fake_response(200, [], content_type="application/json",
                             body_text="{}")
    with mock.patch.object(
        model_adapters.requests, "post", return_value=ok_resp,
    ) as post:
        ready = asyncio.run(adapter.is_model_ready(timeout=2))
    assert ready is True
    # Probe is a tiny non-streaming completion to the chat endpoint.
    assert post.call_args.kwargs["json"]["stream"] is False
    assert post.call_args.kwargs["json"]["max_tokens"] == 1
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"

    for status in (401, 403, 500, 503):
        bad = _fake_response(status, [], content_type="application/json",
                             body_text="{}")
        with mock.patch.object(
            model_adapters.requests, "post", return_value=bad,
        ):
            ready = asyncio.run(adapter.is_model_ready(timeout=2))
        assert ready is False, f"status {status} should be not-ready"
    print("  OK  test_readiness_probe_true_on_200_false_otherwise")


def test_readiness_probe_false_on_exception():
    import asyncio
    adapter = _build_adapter()
    with mock.patch.object(
        model_adapters.requests, "post",
        side_effect=model_adapters.requests.RequestException("boom"),
    ):
        ready = asyncio.run(adapter.is_model_ready(timeout=2))
    assert ready is False
    print("  OK  test_readiness_probe_false_on_exception")


def test_generate_stream_yields_chunks():
    """generate_stream yields each delta incrementally and uses stream=True."""
    adapter = _build_adapter()
    lines = [_sse("Hel"), _sse("lo"), _sse(" world"), "data: [DONE]"]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ) as post:
        got = list(adapter.generate_stream(
            "hi", temperature=0.3, max_new_tokens=64,
        ))
    assert got == ["Hel", "lo", " world"], got
    assert "".join(got) == "Hello world"
    _, kwargs = post.call_args
    assert kwargs["stream"] is True
    assert kwargs["json"]["stream"] is True
    print("  OK  test_generate_stream_yields_chunks")


def test_generate_stream_buffered_fallback_yields_once():
    """A non-SSE (buffered) response yields the recovered text exactly once."""
    adapter = _build_adapter()
    body = json.dumps({"choices": [{"message": {"content": "buffered answer"}}]})
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, [], content_type="application/json",
                                    body_text=body),
    ):
        got = list(adapter.generate_stream(
            "hi", temperature=0.3, max_new_tokens=64,
        ))
    assert got == ["buffered answer"], got
    print("  OK  test_generate_stream_buffered_fallback_yields_once")


def test_generate_stream_non_2xx_raises():
    adapter = _build_adapter()
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(503, []),
    ):
        try:
            list(adapter.generate_stream("hi", temperature=0.3, max_new_tokens=64))
            raised = False
        except RuntimeError as e:
            raised = True
            assert "DeepInfra error 503" in str(e)
    assert raised
    print("  OK  test_generate_stream_non_2xx_raises")


def test_vllm_generate_stream_yields_chunks():
    """The vLLM adapter's generate_stream behaves the same (shared helper)."""
    adapter = model_adapters.VLLMStreamingLLM(
        base_url="https://vllm.example.com", api_key="k",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
    )
    lines = [_sse("a"), _sse("b"), _sse("c"), "data: [DONE]"]
    with mock.patch.object(
        model_adapters.requests, "post",
        return_value=_fake_response(200, lines),
    ):
        got = list(adapter.generate_stream("hi", temperature=0.3, max_new_tokens=64))
    assert got == ["a", "b", "c"], got
    print("  OK  test_vllm_generate_stream_yields_chunks")


def test_rag_controller_resolves_deepinfra_stream():
    """rag_controller.get_model_type('deepinfra_stream') returns the adapter."""
    import os
    os.environ.setdefault("DEEPINFRA_TOKEN", "x")
    # rag_controller pulls in rag_retrieval; only import if available.
    try:
        import rag_controller
    except Exception as e:  # pragma: no cover - environment-dependent
        print(f"  SKIP test_rag_controller_resolves_deepinfra_stream ({e})")
        return
    m = rag_controller.get_model_type("deepinfra_stream")
    assert isinstance(m, model_adapters.DeepInfraStreamingLLM)
    print("  OK  test_rag_controller_resolves_deepinfra_stream")


if __name__ == "__main__":
    tests = [
        test_basic_sse_concatenates_deltas,
        test_done_marker_terminates_stream,
        test_malformed_lines_are_skipped,
        test_non_sse_buffered_fallback,
        test_non_2xx_raises_with_body_excerpt,
        test_connect_error_raises,
        test_headers_include_bearer_and_sse_accept_no_cf,
        test_prevalidate_rejects_empty_prompt_and_missing_token,
        test_payload_system_appears_once_and_stream_true,
        test_payload_omits_system_when_none,
        test_factory_creates_deepinfra_stream,
        test_is_valid_model_type_accepts_deepinfra_stream,
        test_readiness_probe_true_on_200_false_otherwise,
        test_readiness_probe_false_on_exception,
        test_generate_stream_yields_chunks,
        test_generate_stream_buffered_fallback_yields_once,
        test_generate_stream_non_2xx_raises,
        test_vllm_generate_stream_yields_chunks,
        test_rag_controller_resolves_deepinfra_stream,
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
    print(f"{len(tests) - len(failures)} / {len(tests)} deepinfra stream adapter tests passed")
    sys.exit(1 if failures else 0)
