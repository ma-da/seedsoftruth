#!/usr/bin/env python3
"""
Probe a vLLM /v1/chat/completions endpoint with stream=true and time
every SSE chunk it returns.

Purpose
-------
Cloudflare's 100s Proxy Read Timeout only fires when there's a >100s gap
between bytes from the origin. The Seeds of Truth vLLM adapter relies on
SSE chunks flowing continuously to keep the clock at zero. This script
verifies that — end to end, through whatever's between you and the
vLLM server — chunks actually arrive incrementally rather than in one
buffered burst at the end.

If chunks arrive one-per-token over the duration of generation: stream
is healthy and the 524 fix will work. If all chunks arrive in a tight
window at the very end: something is buffering (most often a Cloudflare
proxy setting, or a misconfigured nginx/reverse-proxy in front of vLLM)
and the adapter's streaming behavior will be silently neutralized.

Env vars
--------
Reads with the same SOT_VLLM_* → SPARK_* fallback chain as the live
adapter (see model_adapters.py):

  SOT_VLLM_BASE_URL              or SPARK_BASE_URL
  SOT_VLLM_API_KEY               or SPARK_SITE_API_KEY
  SOT_VLLM_MODEL_NAME            or SPARK_MODEL_NAME
  SOT_VLLM_CF_ACCESS_CLIENT_ID   or SPARK_CF_ACCESS_CLIENT_ID
  SOT_VLLM_CF_ACCESS_CLIENT_SECRET or SPARK_CF_ACCESS_CLIENT_SECRET

CLI overrides
-------------
  --url URL              override base URL
  --model NAME           override served model id
  --prompt TEXT          prompt (default: "Count slowly from 1 to 30.")
  --max-tokens N         cap response length (default 256)
  --connect-timeout SEC  TCP/TLS connect timeout (default 10)
  --read-timeout SEC     max gap between chunks before bailing (default 60)
  --verbose, -v          print every chunk individually
  --quiet, -q            only print the verdict line
  --self-test            run against a synthetic SSE feed (no network)

Examples
--------
  python3 tools/probe_vllm_stream.py
  python3 tools/probe_vllm_stream.py --prompt "Write a haiku about latency"
  python3 tools/probe_vllm_stream.py --url https://other.example.com -v
  python3 tools/probe_vllm_stream.py --self-test    # offline sanity check
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Pretty printing (best-effort ANSI; degrades to plain on non-TTY)
# --------------------------------------------------------------------------
class C:
    """ANSI color constants; empty strings when stdout is not a TTY."""
    USE = sys.stdout.isatty()
    RESET = "\033[0m" if USE else ""
    BOLD = "\033[1m" if USE else ""
    GREY = "\033[90m" if USE else ""
    GREEN = "\033[92m" if USE else ""
    YELLOW = "\033[93m" if USE else ""
    RED = "\033[91m" if USE else ""
    CYAN = "\033[96m" if USE else ""


def paint(s: str, color: str) -> str:
    """Wrap a string in an ANSI color code (no-op when colors are disabled).

    Args:
        s: The text to colorize.
        color: An ANSI escape sequence (e.g. one of the ``C`` constants).

    Returns:
        The colorized string, or plain text if color is off.
    """
    return f"{color}{s}{C.RESET}"


# --------------------------------------------------------------------------
# Env-var chain — mirrors model_adapters._vllm_env_chain()
# --------------------------------------------------------------------------
def env_chain(*keys: str, default: str = "") -> str:
    """Return the first non-empty (stripped) environment variable among ``keys``.

    Args:
        *keys: Environment variable names to check in priority order.
        default: Value to return if none of the variables are set/non-empty.

    Returns:
        The first non-empty value found, else ``default``.
    """
    for k in keys:
        v = os.getenv(k, "").strip()
        if v:
            return v
    return default


def resolve_config(args: argparse.Namespace) -> dict:
    """Resolve endpoint config from CLI args, env-var fallback chain, and defaults.

    Args:
        args: Parsed CLI args (``url`` and ``model`` may override env vars).

    Returns:
        A dict with ``base_url``, ``model``, ``api_key``, ``cf_id`` and
        ``cf_secret`` keys.
    """
    return {
        "base_url": (
            args.url
            or env_chain("SOT_VLLM_BASE_URL", "SPARK_BASE_URL",
                         default="https://seedsoftruth.peerservice.org")
        ).rstrip("/"),
        "model": (
            args.model
            or env_chain("SOT_VLLM_MODEL_NAME", "SPARK_MODEL_NAME",
                         default="wtk_gamma_v9")
        ),
        "api_key": env_chain("SOT_VLLM_API_KEY", "SPARK_SITE_API_KEY"),
        "cf_id": env_chain("SOT_VLLM_CF_ACCESS_CLIENT_ID", "SPARK_CF_ACCESS_CLIENT_ID"),
        "cf_secret": env_chain("SOT_VLLM_CF_ACCESS_CLIENT_SECRET", "SPARK_CF_ACCESS_CLIENT_SECRET"),
    }


# --------------------------------------------------------------------------
# Headers
# --------------------------------------------------------------------------
def build_headers(cfg: dict) -> dict:
    """Build the HTTP request headers for the streaming probe.

    Always sets JSON content type and an SSE Accept header; conditionally adds
    bearer auth and Cloudflare Access headers when present in ``cfg``.

    Args:
        cfg: The config dict from ``resolve_config``.

    Returns:
        A dict of request headers.
    """
    h = {
        "Content-Type": "application/json",
        # Tells Cloudflare and any other proxy that we expect streaming.
        "Accept": "text/event-stream",
    }
    if cfg["api_key"]:
        h["Authorization"] = f"Bearer {cfg['api_key']}"
    if cfg["cf_id"]:
        h["CF-Access-Client-Id"] = cfg["cf_id"]
    if cfg["cf_secret"]:
        h["CF-Access-Client-Secret"] = cfg["cf_secret"]
    return h


# --------------------------------------------------------------------------
# Self-test SSE source (no network) — used by --self-test
# --------------------------------------------------------------------------
def synth_sse_lines(n_chunks: int = 20, gap_secs: float = 0.05):
    """Generator that mimics a vLLM-style SSE stream for offline testing."""
    for i in range(n_chunks):
        time.sleep(gap_secs)
        payload = json.dumps({
            "choices": [{"delta": {"content": f"tok{i} "}, "finish_reason": None}],
        })
        yield f"data: {payload}"
    yield "data: [DONE]"


def synth_buffered_lines(n_chunks: int = 20, gen_secs: float = 5.0):
    """Simulate a buffered stream: dark window, then a burst at the end."""
    time.sleep(gen_secs)
    for i in range(n_chunks):
        payload = json.dumps({
            "choices": [{"delta": {"content": f"tok{i} "}, "finish_reason": None}],
        })
        yield f"data: {payload}"
    yield "data: [DONE]"


def synth_non_streaming_lines(gen_secs: float = 1.0):
    """Simulate a server that ignores stream=true and returns a buffered
    JSON response (no `data: ` prefix). This is the failure mode where
    something between you and vLLM accepts the OpenAI request shape but
    awaits the full reply before sending bytes."""
    time.sleep(gen_secs)
    body = json.dumps({
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "wtk_gamma_v9",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }],
    })
    # iter_lines splits on \n, so a single line is fine here.
    yield body


# --------------------------------------------------------------------------
# Verdict logic
# --------------------------------------------------------------------------
def verdict(
    total_secs: float,
    ttft_secs: float,
    n_content_chunks: int,
    inter_gaps: List[float],
    non_sse_lines: Optional[List[str]] = None,
) -> Tuple[str, str, str]:
    """
    Returns (label, color, explanation).

    Heuristics:
      - n_content_chunks == 0 AND non_sse_lines non-empty → server
        ignored stream=true entirely and returned a buffered JSON body.
        This is a different failure mode from "Cloudflare buffered the
        SSE stream" — the bytes never had SSE structure to begin with.
      - n_content_chunks <= 1 → no streaming evidence (server may have
        returned the whole answer in one chunk; not necessarily broken
        but indistinguishable from buffering).
      - ttft > 75% of total → response was delivered in a tight burst
        at the end; almost certainly buffered.
      - max gap > 30s → real cause for concern; will hit CF timeout on
        longer generations.
      - otherwise → healthy.
    """
    if n_content_chunks == 0 and non_sse_lines:
        # Truncate the captured body for the explanation.
        joined = "\n".join(non_sse_lines)
        preview = joined[:600] + ("..." if len(joined) > 600 else "")
        return (
            "NOT_STREAMING",
            C.RED,
            "The server returned a 200 but the body had NO `data: ...` SSE "
            "chunks at all. It's serving stream=true requests as if they "
            "were non-streaming — most likely a wrapper in front of vLLM "
            "(custom Flask shim, proxy, or load balancer) that accepts the "
            "OpenAI request shape but always buffers the model's output "
            "into a single JSON response. vLLM itself, run with the standard "
            "OpenAI entrypoint, would have emitted SSE here.\n\n"
            f"What the body actually contained (first 600 chars):\n"
            f"  {preview}\n\n"
            "Fix: either point Cloudflare directly at vLLM's OpenAI server "
            "(skip the wrapper), or update the wrapper to forward SSE chunks "
            "from vLLM upstream instead of awaiting the full response.",
        )

    if n_content_chunks <= 1:
        return (
            "INDETERMINATE",
            C.YELLOW,
            "Only one chunk received. The server returned the answer all at "
            "once, which is indistinguishable from a buffered stream. Try a "
            "longer prompt or larger --max-tokens to force a multi-chunk "
            "response.",
        )

    # If 90%+ of total time elapsed before the first chunk, the stream
    # was likely buffered upstream.
    if total_secs > 0 and ttft_secs / total_secs > 0.75:
        return (
            "BUFFERED",
            C.RED,
            f"First chunk took {ttft_secs:.2f}s of a {total_secs:.2f}s total "
            f"response — {ttft_secs/total_secs:.0%} of the response was "
            "delivered after silence. Something between you and vLLM is "
            "buffering. The Seeds of Truth adapter will silently fall back "
            "to non-streaming behavior and continue hitting Cloudflare 524s. "
            "Most likely causes: a Cloudflare Page Rule or Cache Rule "
            "covering /v1/chat/completions, an nginx reverse proxy with "
            "proxy_buffering on, or HTTP/1.0 forced somewhere on the path. "
            "Cloudflare Tunnel (cloudflared) is the cleanest fix.",
        )

    max_gap = max(inter_gaps) if inter_gaps else 0.0
    if max_gap > 30.0:
        return (
            "SLOW",
            C.YELLOW,
            f"Stream is flowing but the largest inter-chunk gap was "
            f"{max_gap:.1f}s. That's not buffering, but it's close enough "
            "to Cloudflare's 100s ceiling that you'll get 524s on long "
            "generations or slow prefill. Enable --enable-chunked-prefill "
            "on vLLM if you haven't already.",
        )

    return (
        "HEALTHY",
        C.GREEN,
        f"Stream is healthy. {n_content_chunks} content chunks delivered "
        f"over {total_secs:.2f}s, median gap "
        f"{statistics.median(inter_gaps):.3f}s. "
        "Cloudflare's 100s read timeout will not fire on this path.",
    )


# --------------------------------------------------------------------------
# Core probe
# --------------------------------------------------------------------------
def probe(
    line_iter,
    *,
    started_at: float,
    headers_at: float,
    args: argparse.Namespace,
) -> int:
    """Iterate SSE lines, time each chunk, print, and emit a verdict."""
    chunk_times: List[float] = []   # timestamp of each content chunk (relative)
    chunk_texts: List[str] = []
    saw_done = False
    last_time = headers_at

    # We also capture any non-data: lines so that when the response body
    # isn't SSE-formatted at all (i.e. the server returned a buffered JSON
    # blob and ignored stream=true), we can surface what we actually
    # received instead of leaving the user staring at "0 chunks".
    non_sse_lines: List[str] = []
    NON_SSE_MAX = 8 * 1024  # capture up to ~8KB before truncating

    for raw in line_iter:
        now = time.time()
        if not raw:
            continue
        if not raw.startswith("data: "):
            # Not an SSE data line. Stash the raw line for diagnostics.
            if sum(len(l) for l in non_sse_lines) < NON_SSE_MAX:
                non_sse_lines.append(raw)
            continue

        payload_str = raw[len("data: "):]
        if payload_str == "[DONE]":
            saw_done = True
            if args.verbose:
                rel = now - started_at
                gap = now - last_time
                print(paint(
                    f"  [{rel:7.3f}s] gap={gap:6.3f}s  [DONE]",
                    C.GREY,
                ))
            break

        try:
            event = json.loads(payload_str)
        except ValueError:
            continue

        choices = event.get("choices") if isinstance(event, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        c0 = choices[0] or {}
        delta = c0.get("delta") or {}
        piece = delta.get("content")
        if not isinstance(piece, str) or not piece:
            continue

        rel = now - started_at
        gap = now - last_time
        chunk_times.append(rel)
        chunk_texts.append(piece)
        last_time = now

        if args.verbose:
            preview = piece.replace("\n", "\\n")
            if len(preview) > 40:
                preview = preview[:37] + "..."
            print(
                f"  [{rel:7.3f}s] gap={gap:6.3f}s  {paint(repr(preview), C.CYAN)}"
            )

    total_secs = time.time() - started_at
    ttft = chunk_times[0] if chunk_times else total_secs
    n_content = len(chunk_times)
    inter_gaps = [
        chunk_times[i] - chunk_times[i - 1] for i in range(1, n_content)
    ]
    full_text = "".join(chunk_texts)

    # --- Summary ---
    if not args.quiet:
        print()
        print(paint("Summary", C.BOLD))
        print("-------")
        print(f"  Total time:           {total_secs:.3f}s")
        print(f"  Time to headers:      {headers_at - started_at:.3f}s")
        print(f"  Time to first chunk:  {ttft:.3f}s")
        print(f"  Content chunks:       {n_content}")
        print(f"  [DONE] sentinel:      {'yes' if saw_done else 'NO ⚠️'}")
        if inter_gaps:
            print(f"  Median inter-chunk:   {statistics.median(inter_gaps):.3f}s")
            print(f"  Mean inter-chunk:     {statistics.mean(inter_gaps):.3f}s")
            print(f"  Min / Max inter-chunk:{min(inter_gaps):.3f}s / {max(inter_gaps):.3f}s")
            # Effective tokens/sec assumes one "chunk" ≈ one token, which
            # is roughly true for vLLM's OpenAI-compat output.
            print(f"  Approx tokens/sec:    {(n_content / total_secs):.1f}")
        if args.verbose:
            print()
            print(paint("Reconstructed reply:", C.GREY))
            print(full_text[:500] + ("..." if len(full_text) > 500 else ""))

    label, color, explanation = verdict(
        total_secs, ttft, n_content, inter_gaps, non_sse_lines=non_sse_lines,
    )
    print()
    print(f"VERDICT: {paint(label, C.BOLD + color)}")
    print(f"  {explanation}")

    # Exit code:
    #   0 = HEALTHY
    #   1 = BUFFERED
    #   2 = SLOW or INDETERMINATE
    #   4 = NOT_STREAMING (server returned non-SSE body)
    if label == "HEALTHY":
        return 0
    if label == "BUFFERED":
        return 1
    if label == "NOT_STREAMING":
        return 4
    return 2


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    """Parse args and run the streaming probe (or a self-test) end to end.

    In ``--self-test`` mode feeds a synthetic SSE stream to ``probe``;
    otherwise resolves config, POSTs a streaming chat request, and times the
    returned SSE chunks.

    Returns:
        Process exit code: 0 HEALTHY, 1 BUFFERED, 2 SLOW/INDETERMINATE, 3 on
        config/connection/HTTP errors, 4 NOT_STREAMING.
    """
    p = argparse.ArgumentParser(
        description="Probe vLLM /v1/chat/completions for streaming behavior.",
    )
    p.add_argument("--url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default="Count slowly from 1 to 30.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--connect-timeout", type=int, default=10)
    p.add_argument("--read-timeout", type=int, default=60)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument(
        "--self-test",
        choices=["healthy", "buffered", "not-streaming"],
        nargs="?",
        const="healthy",
        help="Run against a synthetic feed instead of the network. "
             "Useful for verifying the probe logic before pointing it at "
             "a real endpoint. 'healthy' is the default and emits SSE "
             "chunks evenly. 'buffered' delays then bursts SSE chunks. "
             "'not-streaming' returns a single buffered JSON body — the "
             "failure mode where the server ignores stream=true.",
    )
    args = p.parse_args()

    if args.self_test:
        print(paint(
            f"Running self-test against synthetic SSE ({args.self_test})", C.YELLOW,
        ))
        print("-" * 60)
        started_at = time.time()
        # Pretend headers arrived "immediately" so the timing chain works
        headers_at = started_at + 0.01
        time.sleep(0.01)
        if args.self_test == "buffered":
            lines = synth_buffered_lines(n_chunks=20, gen_secs=2.0)
        elif args.self_test == "not-streaming":
            lines = synth_non_streaming_lines(gen_secs=1.0)
        else:
            lines = synth_sse_lines(n_chunks=20, gap_secs=0.05)
        return probe(lines, started_at=started_at, headers_at=headers_at, args=args)

    cfg = resolve_config(args)

    if not cfg["base_url"]:
        print(paint(
            "ERROR: no base URL configured. Set SOT_VLLM_BASE_URL or pass --url.",
            C.RED,
        ))
        return 3

    endpoint = f"{cfg['base_url']}/v1/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    headers = build_headers(cfg)

    # Banner
    if not args.quiet:
        print(paint("vLLM streaming probe", C.BOLD))
        print(f"  Endpoint:    {endpoint}")
        print(f"  Model:       {cfg['model']}")
        print(f"  Prompt:      {args.prompt!r}")
        print(f"  Max tokens:  {args.max_tokens}")
        print(f"  Auth header: {'Bearer ***' if cfg['api_key'] else 'none'}")
        print(f"  CF Access:   {'yes' if cfg['cf_id'] else 'no'}")
        print()

    # Lazy-import requests so --self-test works in environments without it.
    try:
        import requests
    except ImportError:
        print(paint(
            "ERROR: `requests` is not installed. "
            "pip install requests (or use --self-test).",
            C.RED,
        ))
        return 3

    started_at = time.time()
    try:
        r = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=(args.connect_timeout, args.read_timeout),
            stream=True,
        )
    except requests.RequestException as e:
        print(paint(f"ERROR: connection failed: {e}", C.RED))
        return 3

    headers_at = time.time()

    if not args.quiet:
        print(paint(
            f"POST → {r.status_code} in {headers_at - started_at:.3f}s",
            C.GREEN if r.ok else C.RED,
        ))

    if not r.ok:
        body = ""
        try:
            body = r.text[:2000]
        except Exception:
            pass
        print(paint(f"ERROR: HTTP {r.status_code}: {body}", C.RED))
        try:
            r.close()
        except Exception:
            pass
        return 3

    if args.verbose:
        print()
        print(paint("Per-chunk timings:", C.GREY))

    try:
        return probe(
            r.iter_lines(decode_unicode=True),
            started_at=started_at,
            headers_at=headers_at,
            args=args,
        )
    finally:
        try:
            r.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
