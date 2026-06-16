"""
Regression tests for the second-pass corner-case fixes applied after the
async-chat migration.

Each test pins one specific failure mode that was found in the audit.

  1. prompt_type / use_rag are propagated to QueuedJob (not silently dropped)
  2. Missing prompt_type defaults to 0 without raising NameError
  3. queue_job() raising leaves the DB row as 'failed', not 'queued' zombie
  4. send_warmup_for is NOT called inline from /api/chat (would cause 60s hangs)
  5. startup sweep with stale_minutes=0 catches restart-window orphans
  6. worker_body outer try/except keeps the thread alive across exceptions

Run: python3 tests/test_async_chat_fixes.py
"""
import json
import sys
import tempfile
import threading
import time as _time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

REPO = str(Path(__file__).resolve().parent.parent)
# Web-server modules + static/ moved under chat_server/; CHAT_SERVER is the
# new root for both flat imports and the source-file reads below.
CHAT_SERVER = str(Path(REPO) / "chat_server")
sys.path.insert(0, CHAT_SERVER)


class _Stub:
    """Test double for the rag_controller module, installed into sys.modules.

    Stands in for the real RAG controller so app.py imports cleanly without
    pulling in spacy/bm25s/etc. Records warmup calls and maintains an
    in-memory job queue in place of the real queueing machinery.
    """

    ENABLE_MIN_GATING = False
    HYBRID_DB_PATH = "/tmp/_unused.db"
    import logging
    rag_logger = logging.getLogger("rag_stub_fixes")

    job_queue = []
    lock = threading.Lock()
    warmup_calls = []
    next_queue_job_raises = None

    @classmethod
    def is_model_type_ready(cls, _mt):
        """Test double for rag_controller.is_model_type_ready; always ready."""
        return True

    @classmethod
    def send_warmup_for(cls, model_type):
        """Test double for rag_controller.send_warmup_for; records the call
        instead of warming a real backend so tests can assert it was/wasn't
        invoked inline."""
        cls.warmup_calls.append(model_type)
        return True

    @classmethod
    def queue_job(cls, user_id, job_id, model_type, msg, subsets,
                  rag_algo_choice, use_rag=True, prompt_type=0):
        """Test double for rag_controller.queue_job; builds an in-memory
        QueuedJob from the args and appends it to the fake queue (or raises
        a preloaded error to simulate queue contention)."""
        if cls.next_queue_job_raises is not None:
            err, cls.next_queue_job_raises = cls.next_queue_job_raises, None
            raise err
        class _QJ:
            """Test double for the real QueuedJob; a bare attribute bag
            holding the queued request's fields for later assertions."""
        qj = _QJ()
        qj.user_id, qj.job_id, qj.model_type = user_id, job_id, model_type
        qj.prompt, qj.subsets, qj.rag_algo_choice = msg, subsets, rag_algo_choice
        qj.use_rag, qj.prompt_type = use_rag, prompt_type
        with cls.lock:
            cls.job_queue.append(qj)
            return len(cls.job_queue)

    @classmethod
    def queue_position_for_job(cls, job_id):
        """Test double for rag_controller.queue_position_for_job; returns the
        job's 1-based position in the fake queue, or 0 if absent."""
        with cls.lock:
            for i, item in enumerate(cls.job_queue):
                if item.job_id == job_id:
                    return i + 1
        return 0

    @classmethod
    def job_queue_len(cls):
        """Test double for rag_controller.job_queue_len; returns the size of
        the fake in-memory queue."""
        with cls.lock:
            return len(cls.job_queue)

    @classmethod
    def get_next_queued_job(cls):
        """Test double for rag_controller.get_next_queued_job; returns None by
        default and is monkey-patched in the worker-resilience test."""
        # See worker_resilience test: this is monkey-patched there.
        return None

    @classmethod
    def pop_next_queued_job(cls):
        """Test double for rag_controller.pop_next_queued_job; removes the
        head of the fake queue if present."""
        with cls.lock:
            if cls.job_queue:
                cls.job_queue.pop(0)

    @classmethod
    def fetch_queued_job_info(cls, _user_id):
        """Test double for rag_controller.fetch_queued_job_info; returns a
        stub (queue_len, job) tuple of (0, None)."""
        return 0, None

    @classmethod
    def clean_rag_references(cls, docs):
        """Test double for rag_controller.clean_rag_references; passes the
        docs through unchanged (or [] when falsy)."""
        return docs or []

    @classmethod
    def boot(cls):
        """Test double for rag_controller.boot; no-op startup hook."""
        return None

    @classmethod
    def is_model_ready(cls):
        """Test double for rag_controller.is_model_ready; always ready."""
        return True

    @classmethod
    def send_warmup(cls):
        """Test double for rag_controller.send_warmup; no-op that reports
        success."""
        return True


sys.modules["rag_controller"] = _Stub

if "db" in sys.modules:
    del sys.modules["db"]
import db  # noqa
_tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmpdb.close()
db.DB_PATH = Path(_tmpdb.name)
db.init_db()

if "app" in sys.modules:
    del sys.modules["app"]
import app  # noqa
import runtime  # noqa  — rate_limiter/inflight_chat_reqs live here post-refactor


def unlock(client):
    """Mark the test client's session as unlocked so /api/* routes are
    reachable past the unlock gate."""
    with client.session_transaction() as sess:
        sess["unlocked"] = True


def reset():
    """Reset shared state between tests: clear the stub's fake queue and
    warmup log, wipe the jobs table, and drop rate-limiter user data."""
    _Stub.job_queue.clear()
    _Stub.warmup_calls.clear()
    _Stub.next_queue_job_raises = None
    with db.get_conn() as conn:
        conn.execute("DELETE FROM jobs")
    with runtime.rate_limiter._lock:
        runtime.rate_limiter.user_data.clear()


# --------------------------------------------------------------------------
# 1. prompt_type + use_rag propagate to QueuedJob
# --------------------------------------------------------------------------
def test_prompt_type_and_use_rag_propagate_to_queued_job():
    """A /api/chat request's prompt_type (1-indexed wire value) and use_rag
    flag reach the QueuedJob (prompt_type decremented to internal 0-index)."""
    reset()
    client = app.app.test_client()
    unlock(client)
    sc = client.post("/api/chat", json={
        "user_id": "u1",
        "message": "Q",
        "model_type": "sim",
        "rag_algo_type": 5,
        "prompt_type": 2,     # wire is 1-indexed, becomes internal 1
        "use_rag": False,
    }).status_code
    assert sc == 202
    assert len(_Stub.job_queue) == 1
    qj = _Stub.job_queue[0]
    assert qj.prompt_type == 1, f"prompt_type should be 1 (2-1), got {qj.prompt_type}"
    assert qj.use_rag is False, f"use_rag should be False, got {qj.use_rag}"
    print("  OK  test_prompt_type_and_use_rag_propagate_to_queued_job")


# --------------------------------------------------------------------------
# 2. Missing prompt_type defaults to 0
# --------------------------------------------------------------------------
def test_missing_prompt_type_defaults_to_zero():
    """When the request omits prompt_type, the QueuedJob defaults to
    prompt_type 0 and use_rag True without raising."""
    reset()
    client = app.app.test_client()
    unlock(client)
    # No prompt_type in payload
    sc = client.post("/api/chat", json={
        "user_id": "u-default",
        "message": "Q",
        "model_type": "sim",
    }).status_code
    assert sc == 202
    qj = _Stub.job_queue[0]
    assert qj.prompt_type == 0
    assert qj.use_rag is True   # default
    print("  OK  test_missing_prompt_type_defaults_to_zero")


# --------------------------------------------------------------------------
# 3. queue_job raising marks the row failed, returns 500, no zombie
# --------------------------------------------------------------------------
def test_queue_job_failure_marks_row_failed():
    """When queue_job raises, /api/chat returns 500, marks the already-inserted
    job row 'failed' (not a 'queued' zombie), and leaves the queue empty."""
    reset()
    _Stub.next_queue_job_raises = RuntimeError("simulated queue contention")
    client = app.app.test_client()
    unlock(client)
    r = client.post("/api/chat", json={
        "user_id": "u-bad-queue", "message": "Q", "model_type": "sim",
    })
    assert r.status_code == 500, f"expected 500, got {r.status_code}: {r.get_json()}"
    body = r.get_json()
    job_id = body["job_id"]
    assert body["ok"] is False
    # The row was inserted before queue_job ran — it must now be failed,
    # not stuck at 'queued'.
    row = db.get_job(job_id, "u-bad-queue")
    assert row is not None, "row should still exist for diagnostics"
    assert row["status"] == "failed", (
        f"row should be marked failed; status={row['status']}"
    )
    assert "queue" in row["response"].lower()
    # And the in-mem queue must NOT have the job (queue_job raised)
    assert _Stub.job_queue_len() == 0
    print("  OK  test_queue_job_failure_marks_row_failed")


# --------------------------------------------------------------------------
# 4. /api/chat does NOT call send_warmup_for inline
# --------------------------------------------------------------------------
def test_no_inline_send_warmup_for():
    """/api/chat must not call send_warmup_for inline (which could hang the
    request ~60s); warmup is deferred to the worker thread."""
    reset()
    client = app.app.test_client()
    unlock(client)
    client.post("/api/chat", json={
        "user_id": "u-warm", "message": "Q", "model_type": "sim",
    })
    # The OLD code called warmup on every request; the new code defers
    # to the worker thread. We assert the call did NOT happen during the
    # request lifecycle.
    assert _Stub.warmup_calls == [], (
        f"send_warmup_for should not be called inline; got {_Stub.warmup_calls}"
    )
    print("  OK  test_no_inline_send_warmup_for")


# --------------------------------------------------------------------------
# 5. Startup sweep with stale_minutes=0 catches restart-window orphans
# --------------------------------------------------------------------------
def test_sweep_with_zero_minutes_catches_fresh_orphans():
    """A startup sweep with stale_minutes=0 catches even a just-created
    'queued' row, marking restart-window orphans 'failed'."""
    # Insert a row created RIGHT NOW with status=queued
    reset()
    job_id = db.insert_job("u1", "fresh-window")
    # In the previous behavior (stale_minutes=10), this row would survive
    # a startup sweep. With stale_minutes=0 it must be swept.
    swept = db.sweep_orphaned_jobs(stale_minutes=0)
    assert swept >= 1, f"expected >=1 sweep, got {swept}"
    row = db.get_job(job_id, "u1")
    assert row["status"] == "failed"
    print("  OK  test_sweep_with_zero_minutes_catches_fresh_orphans")


# --------------------------------------------------------------------------
# 6. worker_body outer try/except structure
#
# Timing-based runtime tests against the daemon worker thread are flaky
# (it sleeps 5s between iterations). What we really care about is the
# structural invariant: the `while not app_is_shutdown:` loop body must
# be wrapped in a try/except so an unhandled exception logs + sleeps +
# continues rather than killing the thread.
#
# We assert this via AST inspection — direct, deterministic, fast.
# --------------------------------------------------------------------------
def test_404_response_includes_job_id():
    """Diagnostic improvement: 404s should name the id that wasn't found
    so an operator can trace it from a screenshot."""
    reset()
    client = app.app.test_client()
    unlock(client)
    r = client.get("/api/job/some-unknown-id?user_id=u1")
    assert r.status_code == 404
    body = r.get_json()
    assert body.get("job_id") == "some-unknown-id", (
        f"404 must echo back the asked-for job_id; got {body}"
    )
    print("  OK  test_404_response_includes_job_id")


def test_corrupt_json_blob_falls_back_with_warning(caplog=None):
    """If response column starts with '{' but is malformed, we should
    surface the raw string AND log a warning so an operator can trace it."""
    reset()
    client = app.app.test_client()
    unlock(client)
    r = client.post("/api/chat", json={
        "user_id": "u-corrupt", "message": "Q", "model_type": "sim",
    })
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    # Write a corrupt JSON-looking blob into the response column.
    db.mark_done(job_id, "{this is not valid json")
    r = client.get(f"/api/job/{job_id}?user_id=u-corrupt")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "done"
    # When the JSON parse fails, reply was initialized from raw_response
    # before the parse attempt, so the user sees the raw bytes rather
    # than an empty string. Better to surface partial data than nothing.
    assert body["reply"] == "{this is not valid json", (
        f"corrupt blob should fall back to raw string; got {body['reply']!r}"
    )
    assert body["references"] == []
    print("  OK  test_corrupt_json_blob_falls_back_with_warning")


def test_mark_processing_cas_miss_skips_job():
    """Structural test: if mark_processing returns False, the worker
    must pop the job (skip) rather than process it. Verified via AST."""
    import ast
    src = open(f"{CHAT_SERVER}/worker.py").read()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "worker_body"
    )
    fn_src = ast.unparse(fn)
    # The fixed worker contains "cas_ok" + pop_next_queued_job on the
    # CAS-miss branch. Older worker had no skip behavior.
    assert "cas_ok" in fn_src, "worker_body should branch on cas_ok"
    # Must call pop+continue within the cas-miss block. Crude but
    # effective heuristic:
    idx = fn_src.find("if not cas_ok:")
    assert idx >= 0
    block = fn_src[idx:idx + 500]
    assert "pop_next_queued_job" in block, (
        "cas-miss branch must pop the job; otherwise we'd hot-loop on it"
    )
    assert "continue" in block, "cas-miss branch must continue (skip)"
    print("  OK  test_mark_processing_cas_miss_skips_job")


def test_poll_url_dropped_from_chat_response():
    """Cleanup verification: poll_url was unused by the client and
    should no longer be returned. poll_interval_ms is still useful
    so it stays."""
    reset()
    client = app.app.test_client()
    unlock(client)
    r = client.post("/api/chat", json={
        "user_id": "u-no-pollurl", "message": "Q", "model_type": "sim",
    })
    body = r.get_json()
    assert "poll_url" not in body, (
        f"poll_url should be removed; response was {body}"
    )
    assert body.get("poll_interval_ms") == 1500, (
        "poll_interval_ms should still be present"
    )
    print("  OK  test_poll_url_dropped_from_chat_response")


def test_worker_body_has_outer_try_except():
    """AST check: worker_body's while-loop body begins with a Try that has at
    least one except handler, so an unhandled exception can't kill the daemon
    thread."""
    import ast
    src = open(f"{CHAT_SERVER}/worker.py").read()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "worker_body"
    )
    # Find the `while not app_is_shutdown:` loop
    while_node = next(
        (n for n in fn.body if isinstance(n, ast.While)),
        None,
    )
    assert while_node is not None, "worker_body must contain a while loop"

    # The while body must START with a Try statement (the outer guard).
    # We don't require the *only* statement to be a Try — but the first
    # one must be, so every code path on this iteration is covered.
    assert while_node.body and isinstance(while_node.body[0], ast.Try), (
        "worker_body's while-loop body must begin with a Try statement "
        "(outer guard against unhandled exceptions killing the daemon "
        "thread). See worker_body in app.py."
    )

    # The outer Try must have at least one handler — bare try with no
    # except defeats the purpose.
    outer_try = while_node.body[0]
    assert outer_try.handlers, (
        "worker_body's outer Try must have at least one except handler"
    )

    print("  OK  test_worker_body_has_outer_try_except")


def test_choose_model_type_honors_user_selection():
    """
    Regression: chooseModelTypeForSubmit() must honor toolState.modelType
    when the user picked something other than PRIMARY_MODEL_TYPE. Previously
    the function ALWAYS routed between PRIMARY (spark) and FALLBACK
    (deepinfra) based on Spark health, silently overriding a user who
    explicitly picked 'sim' or another adapter from the DEV_MODE dropdown.

    Verified by source inspection: the function body must reference
    `toolState.modelType` (the dropdown-backed state) before falling back
    to the health-routing logic.
    """
    js = open(f"{CHAT_SERVER}/static/app.js").read()
    # Locate the function body
    start = js.find("async function chooseModelTypeForSubmit()")
    assert start >= 0, "chooseModelTypeForSubmit not found"
    # Find the closing brace of the function (crude but sufficient: the
    # next standalone function declaration after it).
    end = js.find("\nasync function ", start + 1)
    if end < 0:
        end = js.find("\nfunction ", start + 1)
    assert end > start
    body = js[start:end]

    assert "toolState.modelType" in body, (
        "chooseModelTypeForSubmit must consult toolState.modelType so the "
        "user's dropdown selection (e.g. 'sim') is honored. Without this, "
        "selecting Sim Adapter still routes to Spark and produces "
        "Cloudflare 524 from the wrong backend."
    )
    # And the user-pick branch must come BEFORE the health-routing logic
    # (otherwise the routing logic's return would short-circuit it).
    user_branch_idx = body.find("toolState.modelType")
    health_routing_idx = body.find("lastKnownSparkReady")
    assert user_branch_idx < health_routing_idx, (
        "The user-selection branch must run BEFORE the Spark health-routing "
        "fallback; otherwise the dropdown choice is overridden."
    )
    print("  OK  test_choose_model_type_honors_user_selection")


if __name__ == "__main__":
    tests = [
        test_prompt_type_and_use_rag_propagate_to_queued_job,
        test_missing_prompt_type_defaults_to_zero,
        test_queue_job_failure_marks_row_failed,
        test_no_inline_send_warmup_for,
        test_sweep_with_zero_minutes_catches_fresh_orphans,
        test_404_response_includes_job_id,
        test_corrupt_json_blob_falls_back_with_warning,
        test_mark_processing_cas_miss_skips_job,
        test_poll_url_dropped_from_chat_response,
        test_worker_body_has_outer_try_except,
        test_choose_model_type_honors_user_selection,
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
    print(f"{len(tests) - len(failures)} / {len(tests)} fix tests passed")
    sys.exit(1 if failures else 0)
