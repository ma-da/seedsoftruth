"""
End-to-end smoke tests for the async-chat endpoint contracts.

Uses Flask test_client to exercise the real route handlers in app.py.
The heavy rag_controller module is stubbed before import (it pulls in
spacy / bm25s which we don't need to test HTTP plumbing).

What is tested:
  - POST /api/chat returns 202 with job_id + status:"queued"
  - The job is inserted into the DB with status="queued"
  - The job is appended to the in-memory rag_controller.job_queue
  - GET /api/job/<id> returns the right shape for queued / processing /
    done (with JSON blob references) / failed / 404
  - Ownership: wrong user_id gets 404
  - Validation: missing user_id gets 400, invalid model_type gets 400
  - /api/queue is now depth-only (no outgoing_resp)

What is NOT tested here:
  - The real LLM call (uses sim flow / direct DB writes)
  - The worker thread (we directly call mark_done / mark_failed to
    simulate worker completion)
  - Cloudflare / nginx layer

Run: python3 test_async_chat_routes.py
"""
import json
import sys
import tempfile
import threading
from pathlib import Path
from unittest import mock

REPO = str(Path(__file__).resolve().parent.parent)
# Web-server modules moved under chat_server/ but import each other flat.
sys.path.insert(0, str(Path(REPO) / "chat_server"))


# --------------------------------------------------------------------------
# Stub rag_controller BEFORE app.py imports it. We provide enough surface
# for app.py's chat + job routes to work; the worker thread is patched
# out separately below so it never runs.
# --------------------------------------------------------------------------
class _StubRagController:
    """Hand-rolled stub mirroring the real rag_controller's queue API."""
    ENABLE_MIN_GATING = False
    HYBRID_DB_PATH = "/tmp/_unused_hybrid.db"
    import logging
    rag_logger = logging.getLogger("rag_stub")

    job_queue = []
    job_lock = threading.Lock()
    model_ready_responses = {}  # model_type -> bool

    @classmethod
    def is_model_type_ready(cls, model_type):
        """Test-double readiness check; returns the value preconfigured in
        ``model_ready_responses`` for ``model_type`` (defaults to True)."""
        return cls.model_ready_responses.get(model_type, True)

    @classmethod
    def send_warmup_for(cls, model_type):
        """Test-double warmup trigger; always succeeds (no real model)."""
        return True

    @classmethod
    def queue_job(cls, user_id, job_id, model_type, msg, subsets,
                  rag_algo_choice, use_rag=True, prompt_type=0):
        """Test-double enqueue; appends a fake queued-job object to the
        in-memory ``job_queue`` and returns its 1-based queue depth."""
        class _QJ:
            """Lightweight stand-in for the real queued-job dataclass,
            holding only the attributes the route handlers read."""
            pass
        qj = _QJ()
        qj.user_id = user_id
        qj.job_id = job_id
        qj.model_type = model_type
        qj.prompt = msg
        qj.subsets = subsets
        qj.rag_algo_choice = rag_algo_choice
        qj.use_rag = use_rag
        qj.prompt_type = prompt_type
        with cls.job_lock:
            cls.job_queue.append(qj)
            return len(cls.job_queue)

    @classmethod
    def queue_position_for_job(cls, job_id):
        """Test-double lookup; returns the 1-based position of ``job_id`` in
        the in-memory queue, or 0 if it is not present."""
        with cls.job_lock:
            for i, item in enumerate(cls.job_queue):
                if item.job_id == job_id:
                    return i + 1
        return 0

    @classmethod
    def job_queue_len(cls):
        """Test-double accessor; returns the current in-memory queue depth."""
        with cls.job_lock:
            return len(cls.job_queue)

    @classmethod
    def pop_next_queued_job(cls):
        """Test-double dequeue; removes the head of the in-memory queue if any."""
        with cls.job_lock:
            if cls.job_queue:
                cls.job_queue.pop(0)

    @classmethod
    def get_next_queued_job(cls):
        """Test-double; always returns None to keep the worker thread idle."""
        # Always returns None so the daemon worker thread (which we can't
        # easily prevent from starting) stays idle, leaving the queue
        # contents available for test assertions.
        return None

    @classmethod
    def boot(cls):
        """Test-double for the real retrieval boot; a no-op returning None so
        init_state() skips retrieval setup the route tests don't need."""
        # init_state() in app.py calls rag_controller.boot(). Returning
        # None lets init_state log a warning but skip retrieval setup —
        # we don't need retrieval for these route tests.
        return None

    @classmethod
    def fetch_queued_job_info(cls, user_id):
        """Test-double; returns a fixed (queued_count, next_job) of (0, None)."""
        return 0, None

    @classmethod
    def clean_rag_references(cls, docs):
        """Test-double passthrough; returns ``docs`` unchanged (or [] if falsy)."""
        return docs or []

    @classmethod
    def is_model_ready(cls):
        """Test-double; reports the model as always ready."""
        return True

    @classmethod
    def send_warmup(cls):
        """Test-double warmup trigger; always succeeds (no real model)."""
        return True


# Install the stub before app imports
sys.modules["rag_controller"] = _StubRagController
# rag_controller exports ENABLE_MIN_GATING + rag_logger via `from ... import`;
# Python evaluates `from rag_controller import X` against the module object
# in sys.modules, so attribute access on _StubRagController is enough.


# --------------------------------------------------------------------------
# Redirect db.DB_PATH to a tempfile and re-init schema.
# --------------------------------------------------------------------------
if "db" in sys.modules:
    del sys.modules["db"]
import db
_tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmpdb.close()
db.DB_PATH = Path(_tmpdb.name)
db.init_db()


# --------------------------------------------------------------------------
# Now safe to import app. Patch init_state + init_worker to no-ops so we
# don't spin up retrieval or background threads during import.
# --------------------------------------------------------------------------
if "app" in sys.modules:
    del sys.modules["app"]
import app  # noqa: E402  — module-level init_state / init_worker run here
import runtime  # noqa: E402  — rate_limiter lives here post-refactor

# Force any worker thread that did spawn to be a no-op: we set a sentinel
# the worker_body checks. The simpler approach: just don't trust the thread
# to run. We simulate completion via direct DB writes.
print(f"Imported app.py — {len(app.app.url_map._rules)} routes registered")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def unlock_session(client, password=None):
    """Bypass the auth gate so chat routes accept us."""
    # The real unlock route validates a password. For tests we set the
    # session directly using the Flask test_client session_transaction.
    with client.session_transaction() as sess:
        sess["unlocked"] = True
        sess["user_id"] = "test-user-1"


def parse(resp):
    """Test helper; unpack a Flask test-client response into
    ``(status_code, parsed_json_body)``."""
    return resp.status_code, resp.get_json()


def reset_state():
    """Clear stub queue + truncate jobs table + reset the per-user rate
    limiter between tests. The real rate limiter has a 30s cooldown that
    would otherwise block back-to-back POSTs from the same user_id."""
    _StubRagController.job_queue.clear()
    _StubRagController.model_ready_responses.clear()
    with db.get_conn() as conn:
        conn.execute("DELETE FROM jobs")
    # Clear in-process rate-limit state. The lock is per-instance so we
    # acquire it before mutating user_data.
    with runtime.rate_limiter._lock:
        runtime.rate_limiter.user_data.clear()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_chat_returns_202_with_job_id():
    """POST /api/chat returns 202 with ok, status="queued", a job_id, the
    echoed user_id, poll_interval_ms, a queue_position >= 1, and a valid
    queue_reason."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "test-user-1",
        "message": "What is contested truth?",
        "model_type": "sim",
        "rag_algo_type": 5,
        "prompt_type": 1,
    }))
    assert sc == 202, f"expected 202, got {sc} body={data}"
    assert data["ok"] is True
    assert data["status"] == "queued"
    assert data["job_id"], "missing job_id"
    assert data["user_id"] == "test-user-1"
    assert data["poll_interval_ms"] == 1500
    assert data["queue_position"] >= 1
    assert data["queue_reason"] in ("queue_busy", "model_warming")
    print("  OK  test_chat_returns_202_with_job_id")
    return data["job_id"]


def test_chat_persists_row_and_enqueues():
    """POST /api/chat inserts a DB row with status="queued" and appends the
    job to the in-memory rag_controller queue."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "test-user-1",
        "message": "Q1",
        "model_type": "sim",
    }))
    assert sc == 202
    job_id = data["job_id"]
    # DB row exists with status=queued
    row = db.get_job(job_id, "test-user-1")
    assert row is not None and row["status"] == "queued"
    # In-mem queue has it
    assert _StubRagController.job_queue_len() == 1
    assert _StubRagController.job_queue[0].job_id == job_id
    print("  OK  test_chat_persists_row_and_enqueues")


def test_chat_validation():
    """POST /api/chat returns 400 when any required field (message, user_id,
    or model_type) is missing."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    # missing message
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u", "model_type": "sim",
    }))
    assert sc == 400
    # missing user_id
    sc, data = parse(client.post("/api/chat", json={
        "message": "hi", "model_type": "sim",
    }))
    assert sc == 400
    # missing model_type
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u", "message": "hi",
    }))
    assert sc == 400
    print("  OK  test_chat_validation")


def test_job_status_queued():
    """GET /api/job/<id> for a freshly queued job returns 200 with
    status="queued" and queue_position == 1."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u1", "message": "Q", "model_type": "sim",
    }))
    job_id = data["job_id"]
    sc, data = parse(client.get(f"/api/job/{job_id}?user_id=u1"))
    assert sc == 200, f"expected 200, got {sc} body={data}"
    assert data["status"] == "queued"
    assert data["queue_position"] == 1
    print("  OK  test_job_status_queued")


def test_job_status_processing():
    """GET /api/job/<id> returns 200 with status="processing" once the job
    has been marked processing in the DB."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u1", "message": "Q", "model_type": "sim",
    }))
    job_id = data["job_id"]
    # Simulate worker flipping to processing
    assert db.mark_processing(job_id) is True
    sc, data = parse(client.get(f"/api/job/{job_id}?user_id=u1"))
    assert sc == 200
    assert data["status"] == "processing"
    print("  OK  test_job_status_processing")


def test_job_status_done_parses_json_blob():
    """GET /api/job/<id> for a done job whose result is a JSON blob returns
    200 with status="done", the parsed reply, and the references list."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u1", "message": "Q", "model_type": "sim",
    }))
    job_id = data["job_id"]
    # Simulate worker completing with the new JSON blob shape
    db.mark_processing(job_id)
    db.mark_done(job_id, json.dumps({
        "reply": "Here is your answer.",
        "references": [{"title": "Doc A", "url": "https://example.com/a"}],
    }))
    sc, data = parse(client.get(f"/api/job/{job_id}?user_id=u1"))
    assert sc == 200
    assert data["status"] == "done"
    assert data["reply"] == "Here is your answer."
    assert isinstance(data["references"], list) and len(data["references"]) == 1
    assert data["references"][0]["title"] == "Doc A"
    print("  OK  test_job_status_done_parses_json_blob")


def test_job_status_done_legacy_plain_string():
    """GET /api/job/<id> for a legacy done job stored as a plain string
    returns 200 with status="done", the string as reply, and empty
    references."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u1", "message": "Q", "model_type": "sim",
    }))
    job_id = data["job_id"]
    # Pre-migration row: plain-string reply, no JSON
    db.mark_processing(job_id)
    db.mark_done(job_id, "Legacy plain answer without JSON")
    sc, data = parse(client.get(f"/api/job/{job_id}?user_id=u1"))
    assert sc == 200
    assert data["status"] == "done"
    assert data["reply"] == "Legacy plain answer without JSON"
    assert data["references"] == []
    print("  OK  test_job_status_done_legacy_plain_string")


def test_job_status_failed():
    """GET /api/job/<id> for a failed job returns 200 with status="failed",
    ok=False, and the failure detail surfaced."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u1", "message": "Q", "model_type": "sim",
    }))
    job_id = data["job_id"]
    db.mark_failed(job_id, "Boom: model timeout")
    sc, data = parse(client.get(f"/api/job/{job_id}?user_id=u1"))
    assert sc == 200
    assert data["status"] == "failed"
    assert data["ok"] is False
    assert "Boom" in data["detail"]
    print("  OK  test_job_status_failed")


def test_job_status_404_wrong_user():
    """GET /api/job/<id> with a user_id that does not own the job returns
    404, never leaking the job's existence to other users."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u1", "message": "Q", "model_type": "sim",
    }))
    job_id = data["job_id"]
    # Wrong user — must NOT leak existence
    sc, data = parse(client.get(f"/api/job/{job_id}?user_id=evil"))
    assert sc == 404
    print("  OK  test_job_status_404_wrong_user")


def test_job_status_404_unknown_id():
    """GET /api/job/<id> for a job id that does not exist returns 404."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.get("/api/job/does-not-exist?user_id=u1"))
    assert sc == 404
    print("  OK  test_job_status_404_unknown_id")


def test_job_status_missing_user_id():
    """GET /api/job/<id> without a user_id query param returns 400."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.get("/api/job/some-id"))
    assert sc == 400
    print("  OK  test_job_status_missing_user_id")


def test_queue_endpoint_is_depth_only():
    """POST /api/queue returns 200 reporting only queue depth
    (queries_in_line) with resps_in_line==0 and no retired outgoing_resp
    field."""
    reset_state()
    client = app.app.test_client()
    unlock_session(client)
    # Fire a couple of chats so queue depth is non-zero
    client.post("/api/chat", json={"user_id": "u1", "message": "a", "model_type": "sim"})
    client.post("/api/chat", json={"user_id": "u2", "message": "b", "model_type": "sim"})
    sc, data = parse(client.post("/api/queue", json={"user_id": "u1"}))
    assert sc == 200
    assert data["ok"] is True
    assert data["queries_in_line"] == 2
    assert data["resps_in_line"] == 0  # outgoing queue retired
    assert "outgoing_resp" not in data, "outgoing_resp must be retired"
    print("  OK  test_queue_endpoint_is_depth_only")


def test_chat_unauthenticated_is_blocked():
    """POST /api/chat without an unlocked session is blocked by the auth
    gate with 401 or 403."""
    reset_state()
    client = app.app.test_client()
    # No unlock_session — gate should block
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u1", "message": "Q", "model_type": "sim",
    }))
    assert sc in (401, 403), f"expected gate to block, got {sc}"
    print("  OK  test_chat_unauthenticated_is_blocked")


def test_queue_reason_reflects_model_readiness():
    """POST /api/chat sets queue_reason to "model_warming" when the model
    type is not ready and "queue_busy" when it is."""
    reset_state()
    _StubRagController.model_ready_responses["sim"] = False
    client = app.app.test_client()
    unlock_session(client)
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u-cold", "message": "Q", "model_type": "sim",
    }))
    assert sc == 202, f"cold POST got {sc}: {data}"
    assert data["queue_reason"] == "model_warming"
    # And again with ready — distinct user_id to avoid the 30s rate-limit cooldown.
    _StubRagController.model_ready_responses["sim"] = True
    sc, data = parse(client.post("/api/chat", json={
        "user_id": "u-warm", "message": "Q2", "model_type": "sim",
    }))
    assert sc == 202, f"warm POST got {sc}: {data}"
    assert data["queue_reason"] == "queue_busy"
    print("  OK  test_queue_reason_reflects_model_readiness")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_chat_returns_202_with_job_id,
        test_chat_persists_row_and_enqueues,
        test_chat_validation,
        test_job_status_queued,
        test_job_status_processing,
        test_job_status_done_parses_json_blob,
        test_job_status_done_legacy_plain_string,
        test_job_status_failed,
        test_job_status_404_wrong_user,
        test_job_status_404_unknown_id,
        test_job_status_missing_user_id,
        test_queue_endpoint_is_depth_only,
        test_chat_unauthenticated_is_blocked,
        test_queue_reason_reflects_model_readiness,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            import traceback as _tb
            print(f"  FAIL  {t.__name__}: {e}")
            _tb.print_exc()
    print()
    print(f"{len(tests) - len(failures)} / {len(tests)} route tests passed")
    sys.exit(1 if failures else 0)
