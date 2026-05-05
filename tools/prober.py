#!/usr/bin/env python3
"""
Seeds of Truth chat prober.

Round-robins a fixed list of queries against /api/chat at a configurable interval,
tracks successes/failures/latency, alerts loudly on the console for any failure,
and writes a timestamped log file alongside the script.

Defaults match the production wire defaults pulled from static/app.js:
  - model_type:     "spark"
  - rag_algo_type:  4   (CFG.DEFAULT_RAG_ALGO_TYPE)
  - prompt_type:    1   (CFG.DEFAULT_PROMPT_TYPE)

Auth: reads SOT_PASSWORD from the environment, or pass --password.

Usage:
  SOT_PASSWORD=... python3 prober.py
  python3 prober.py --url https://seedsoftruth.peerservice.org --interval 180
  python3 prober.py --url http://localhost:5000 --max-iterations 10

Exits non-zero on auth failure or unrecoverable connection error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import signal
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    sys.stderr.write("This script requires `requests`. Install with: pip install requests\n")
    sys.exit(2)


# ---- ANSI color helpers (no external deps) ---------------------------------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}" if supports_color() else text


# ---- Query list (round-robin) ----------------------------------------------

QUERY_LIST = [
    "Tell me about glyphosate cancer Monsanto lawsuit",
    "Tell me about CIA covert operations secrecy",
    "tell me about death squads in Haiti",
    "Describe the activities and background of David Ferrie. Why did investigators view him as an important figure in the New Orleans context surrounding the Kennedy assassination?",
    "Do vaccines cause autism or cancer?",
    "Did Jeffrey Epstein have connections to the cia or mossad?",
    "what is the relationship between cointelpro and the cia project chaos?",
    "is there good evidence of ufo uap et visitation?",
    "at what temperature does cadmium boil?",
    "what is the best region in Switzerland for skiing",
    "what is the airspeed velocity of a swallow",
    "how long does it take to rig a schooner",
    "Who killed JFK?",
    "Who killed Gandhi?",
    "Is psychic phenomena real?",
    "How can we reform government?",
    "How did Building 7 fall during 9/11?",
]


# ---- Defaults pulled from static/app.js ------------------------------------

# use this to test prod
#DEFAULT_MODEL_TYPE = "spark"

# Use this for echo testing
DEFAULT_MODEL_TYPE = "sim"

DEFAULT_RAG_ALGO_TYPE = 4   # static/app.js CFG.DEFAULT_RAG_ALGO_TYPE
DEFAULT_PROMPT_TYPE = 1     # static/app.js CFG.DEFAULT_PROMPT_TYPE


# ---- Prober ----------------------------------------------------------------

class Prober:
    def __init__(self, args: argparse.Namespace, log: logging.Logger):
        self.args = args
        self.log = log
        self.session = requests.Session()
        self.user_id: Optional[str] = None

        # counters
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.latencies_s: list[float] = []
        self.failure_records: list[Tuple[str, str, str]] = []  # (iso_ts, query, reason)

        self._stop = False

    def stop(self, *_a):
        self._stop = True

    # -- HTTP helpers --------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: float) -> requests.Response:
        url = self.args.url.rstrip("/") + path
        return self.session.post(url, json=payload, timeout=timeout)

    def unlock(self) -> None:
        """Authenticate against /api/unlock. Sets self.user_id on success."""
        self.log.info(f"Unlocking against {self.args.url} ...")
        try:
            r = self._post("/api/unlock", {"password": self.args.password}, timeout=15)
        except requests.RequestException as e:
            self.log.error(f"Unlock request failed: {e}")
            raise SystemExit(2)

        if r.status_code != 200:
            self.log.error(f"Unlock rejected: HTTP {r.status_code} {r.text[:200]}")
            raise SystemExit(2)

        data = r.json()
        if not data.get("ok"):
            self.log.error(f"Unlock returned ok=false: {data}")
            raise SystemExit(2)

        # Server returns a fresh user_id per unlock; honor a CLI override if set.
        self.user_id = self.args.user_id or data.get("user_id") or str(uuid.uuid4())
        self.log.info(f"Unlocked. user_id={self.user_id}")

    # -- One probe iteration -------------------------------------------------

    def _build_chat_payload(self, query: str) -> dict:
        return {
            "user_id": self.user_id,
            "message": query,
            "model_type": self.args.model_type,
            "rag_algo_type": self.args.rag_algo_type,
            "prompt_type": self.args.prompt_type,
            "use_rag": True,
            # subsets omitted -> server treats as None
        }

    def _poll_for_queued_response(self, job_id: str, started_at: float) -> Tuple[bool, str]:
        """
        After a 503 (queued) response from /api/chat, poll /api/queue until the
        outgoing_resp arrives or we hit --queue-timeout. Returns (ok, detail).
        """
        deadline = started_at + self.args.queue_timeout
        poll_every = self.args.poll_interval

        while not self._stop and time.time() < deadline:
            time.sleep(poll_every)
            try:
                r = self._post("/api/queue", {"user_id": self.user_id}, timeout=15)
            except requests.RequestException as e:
                return False, f"queue poll network error: {e}"

            if r.status_code != 200:
                return False, f"queue poll HTTP {r.status_code}: {r.text[:200]}"

            data = r.json()
            raw = data.get("outgoing_resp")
            if not raw:
                # still waiting; show queue depth at debug
                qd = data.get("queries_in_line")
                self.log.debug(f"  still queued: queries_in_line={qd}")
                continue

            # outgoing_resp is a JSON string per QueuedResponse.to_json()
            if isinstance(raw, str):
                try:
                    import json
                    resp = json.loads(raw)
                except Exception as e:
                    return False, f"could not parse outgoing_resp: {e}"
            elif isinstance(raw, dict):
                resp = raw
            else:
                return False, f"unexpected outgoing_resp type: {type(raw).__name__}"

            if resp.get("job_id") != job_id:
                # stale / different job; keep waiting
                self.log.debug(f"  outgoing job_id mismatch ({resp.get('job_id')} != {job_id}); waiting")
                continue

            if resp.get("ok"):
                return True, "queued -> success"
            return False, f"queued response error: {resp.get('error')!r} detail={resp.get('detail')!r}"

        if self._stop:
            return False, "stopped during queue poll"
        return False, f"queue poll timed out after {self.args.queue_timeout:.0f}s"

    def probe_once(self, query: str) -> None:
        self.attempts += 1
        preview = (query[:80] + "...") if len(query) > 80 else query
        self.log.info(paint(f"[{self.attempts}] -> {preview}", C.CYAN))

        payload = self._build_chat_payload(query)
        started = time.time()

        try:
            r = self._post("/api/chat", payload, timeout=self.args.chat_timeout)
        except requests.RequestException as e:
            self._record_failure(query, f"chat network error: {e}", started)
            return

        if r.status_code == 200:
            data = r.json()
            if not data.get("ok"):
                self._record_failure(query, f"200 with ok=false: {data.get('error')!r}", started)
                return
            self._record_success(query, started, "direct")
            return

        if r.status_code == 503:
            # queued path
            try:
                data = r.json()
            except Exception:
                self._record_failure(query, f"503 non-JSON body: {r.text[:200]}", started)
                return

            job_id = data.get("job_id")
            if not job_id:
                self._record_failure(query, f"503 with no job_id: {data}", started)
                return

            self.log.info(f"  queued job_id={job_id}, polling /api/queue...")
            ok, detail = self._poll_for_queued_response(job_id, started)
            if ok:
                self._record_success(query, started, detail)
            else:
                self._record_failure(query, detail, started)
            return

        # any other status code is a failure
        self._record_failure(query, f"unexpected HTTP {r.status_code}: {r.text[:200]}", started)

    # -- Bookkeeping ---------------------------------------------------------

    def _record_success(self, query: str, started: float, mode: str) -> None:
        latency = time.time() - started
        self.latencies_s.append(latency)
        self.successes += 1
        self.log.info(
            paint(f"  OK  ({mode})  latency={latency:.2f}s", C.GREEN)
        )

    def _record_failure(self, query: str, reason: str, started: float) -> None:
        latency = time.time() - started
        self.failures += 1
        ts_iso = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self.failure_records.append((ts_iso, query, reason))
        # Loud console alert (BOLD RED, with bell)
        bell = "\a" if sys.stdout.isatty() else ""
        self.log.error(
            paint(
                f"{bell}[ALERT] FAILURE @ {ts_iso} latency={latency:.2f}s :: {reason}",
                C.BOLD + C.RED,
            )
        )
        self.log.error(paint(f"        query: {query}", C.RED))

    def print_running_stats(self) -> None:
        if not self.attempts:
            return
        rate = (self.successes / self.attempts) * 100.0
        if self.latencies_s:
            avg = statistics.mean(self.latencies_s)
            p50 = statistics.median(self.latencies_s)
            mx = max(self.latencies_s)
            mn = min(self.latencies_s)
            lat = f"avg={avg:.2f}s p50={p50:.2f}s min={mn:.2f}s max={mx:.2f}s"
        else:
            lat = "no successful samples"
        self.log.info(
            paint(
                f"-- stats: attempts={self.attempts} ok={self.successes} fail={self.failures} "
                f"({rate:.1f}% ok)  latency: {lat}",
                C.GREY,
            )
        )

    # -- Main loop -----------------------------------------------------------

    def run(self) -> None:
        self.unlock()

        i = 0
        max_iters = self.args.max_iterations
        while not self._stop:
            if max_iters is not None and i >= max_iters:
                self.log.info(f"Reached max-iterations={max_iters}; stopping.")
                break

            query = QUERY_LIST[i % len(QUERY_LIST)]
            self.probe_once(query)
            self.print_running_stats()

            i += 1
            if max_iters is not None and i >= max_iters:
                continue
            if self._stop:
                break

            # Sleep --interval, but in small chunks so Ctrl-C is responsive
            remaining = self.args.interval
            while remaining > 0 and not self._stop:
                step = min(1.0, remaining)
                time.sleep(step)
                remaining -= step

        self._final_summary()

    def _final_summary(self) -> None:
        self.log.info("")
        self.log.info(paint("===== Prober summary =====", C.BOLD))
        self.print_running_stats()
        if self.failure_records:
            self.log.info(paint(f"Failures ({len(self.failure_records)}):", C.RED))
            for ts_iso, q, reason in self.failure_records:
                preview = (q[:70] + "...") if len(q) > 70 else q
                self.log.info(paint(f"  {ts_iso}  {reason}  ::  {preview}", C.RED))
        else:
            self.log.info(paint("No failures recorded.", C.GREEN))


# ---- CLI / logging setup ---------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seeds of Truth chat prober (round-robins QUERY_LIST against /api/chat).",
    )
    p.add_argument(
        "--url",
        default=os.environ.get("SOT_URL", "http://localhost:5000"),
        help="Base URL of the service (default: http://localhost:5000, or $SOT_URL).",
    )
    p.add_argument(
        "--password",
        default=os.environ.get("SOT_PASSWORD"),
        help="Unlock password. Defaults to $SOT_PASSWORD.",
    )
    p.add_argument(
        "--interval", type=float, default=120.0,
        help="Seconds between probes (default: 120).",
    )
    p.add_argument(
        "--max-iterations", type=int, default=None,
        help="Stop after this many probes (default: run until Ctrl-C).",
    )
    p.add_argument(
        "--model-type", default=DEFAULT_MODEL_TYPE,
        help=f"Model adapter to test (default: {DEFAULT_MODEL_TYPE}).",
    )
    p.add_argument(
        "--rag-algo-type", type=int, default=DEFAULT_RAG_ALGO_TYPE,
        help=f"rag_algo_type wire value (default: {DEFAULT_RAG_ALGO_TYPE}).",
    )
    p.add_argument(
        "--prompt-type", type=int, default=DEFAULT_PROMPT_TYPE,
        help=f"prompt_type wire value, 1-indexed (default: {DEFAULT_PROMPT_TYPE}).",
    )
    p.add_argument(
        "--user-id", default=None,
        help="Override user_id (default: use the one returned by /api/unlock).",
    )
    p.add_argument(
        "--chat-timeout", type=float, default=240.0,
        help="HTTP timeout for /api/chat (default: 240s).",
    )
    p.add_argument(
        "--queue-timeout", type=float, default=600.0,
        help="Max seconds to wait for a queued job's response via /api/queue (default: 600).",
    )
    p.add_argument(
        "--poll-interval", type=float, default=5.0,
        help="Seconds between /api/queue polls when a job is queued (default: 5).",
    )
    p.add_argument(
        "--log-file", default=None,
        help="Path to log file (default: ./prober_<ts>.log next to script).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console + file log level (default: INFO).",
    )
    return p


def setup_logging(log_path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("sot.prober")
    logger.setLevel(getattr(logging, level))

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (no color codes stripped; the handler writes whatever we pass).
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — strip ANSI for cleanliness.
    class StripAnsiFormatter(logging.Formatter):
        import re as _re
        _ansi = _re.compile(r"\x1b\[[0-9;]*m")

        def format(self, record):
            s = super().format(record)
            return self._ansi.sub("", s)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, level))
    fh.setFormatter(StripAnsiFormatter(
        fmt="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.password:
        sys.stderr.write(
            "ERROR: no password provided. Set SOT_PASSWORD or pass --password.\n"
        )
        return 2

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(__file__).resolve().parent / f"prober_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log = setup_logging(log_path, args.log_level)
    log.info(paint(
        f"Seeds of Truth prober starting. url={args.url} interval={args.interval}s "
        f"model={args.model_type} rag_algo_type={args.rag_algo_type} prompt_type={args.prompt_type}",
        C.BOLD,
    ))
    log.info(f"Logging to {log_path}")

    prober = Prober(args, log)

    # graceful shutdown on Ctrl-C / SIGTERM
    signal.signal(signal.SIGINT, prober.stop)
    signal.signal(signal.SIGTERM, prober.stop)

    try:
        prober.run()
    except SystemExit:
        raise
    except Exception as e:
        log.exception(f"Prober crashed: {e}")
        return 1
    return 0 if prober.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
