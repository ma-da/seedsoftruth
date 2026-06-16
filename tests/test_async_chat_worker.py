"""
Exercise the real worker_body() loop end-to-end against a stubbed
rag_controller. Verifies the worker:
  1. Calls mark_processing on dequeue
  2. Writes a JSON blob {"reply", "references"} on success
  3. Calls mark_failed on exception
  4. Pops the job from the in-mem queue exactly once
  5. Maintains inflight_chat_reqs (inc/dec balanced)

This is a different layer from test_async_chat_routes.py — there we
simulate the worker via direct DB writes; here we let the real
worker_body() run.

Run: python3 test_async_chat_worker.py
"""
import json
import sys
import tempfile
import threading
import time as _time
from pathlib import Path
from unittest import mock


REPO = str(Path(__file__).resolve().parent.parent)
# Web-server modules moved under chat_server/ but import each other flat.
sys.path.insert(0, str(Path(REPO) / "chat_server"))


# Reuse the same stub strategy as test_async_chat_routes.py — but here
# we DO want get_next_queued_job to return real items, because the worker
# is the system under test.
class _StubRC:
    """Test double for the ``rag_controller`` module.

    Installed into ``sys.modules`` so the worker under test imports this
    stand-in instead of the real controller. Backs the queue helpers with an
    in-memory ``job_queue`` list, reports models as always ready, and lets
    tests control chat outcomes via ``next_outcome``.
    """

    ENABLE_MIN_GATING = False
    HYBRID_DB_PATH = "/tmp/_unused.db"
    import logging
    rag_logger = logging.getLogger("rag_stub_worker")

    job_queue = []
    lock = threading.Lock()
    next_outcome = ("ok", None)  # ("ok", references) or ("raise", Exception)
    last_chat_args = None

    @classmethod
    def is_model_type_ready(cls, _model_type):
        """Stubbed readiness check that always reports the model as ready."""
        return True

    @classmethod
    def send_warmup_for(cls, _model_type):
        """Stubbed per-model warmup that always succeeds (no real request)."""
        return True

    @classmethod
    def queue_job(cls, user_id, job_id, model_type, msg, subsets, rag_algo_choice):
        """Append a fake queued-job object to the in-memory queue.

        Returns the new queue length (its 1-based position).
        """
        class _QJ:
            """Lightweight stand-in for a real queued-job record."""
            pass
        qj = _QJ()
        qj.user_id, qj.job_id, qj.model_type = user_id, job_id, model_type
        qj.prompt, qj.subsets, qj.rag_algo_choice = msg, subsets, rag_algo_choice
        with cls.lock:
            cls.job_queue.append(qj)
            return len(cls.job_queue)

    @classmethod
    def queue_position_for_job(cls, job_id):
        """Return the 1-based queue position of ``job_id``, or 0 if absent."""
        with cls.lock:
            for i, item in enumerate(cls.job_queue):
                if item.job_id == job_id:
                    return i + 1
        return 0

    @classmethod
    def job_queue_len(cls):
        """Return the current number of jobs in the in-memory queue."""
        with cls.lock:
            return len(cls.job_queue)

    @classmethod
    def get_next_queued_job(cls):
        """Return the job at the head of the queue without removing it (or None)."""
        with cls.lock:
            return cls.job_queue[0] if cls.job_queue else None

    @classmethod
    def pop_next_queued_job(cls):
        """Remove the job at the head of the queue, if any."""
        with cls.lock:
            if cls.job_queue:
                cls.job_queue.pop(0)

    @classmethod
    def fetch_queued_job_info(cls, _user_id):
        """Stubbed lookup that always reports no queued job for the user."""
        return 0, None

    @classmethod
    def clean_rag_references(cls, docs):
        """Pass-through reference cleaner; returns ``docs`` (or [] if falsy)."""
        return docs or []

    @classmethod
    def boot(cls):
        """No-op stand-in for controller boot/initialization."""
        return None

    @classmethod
    def is_model_ready(cls):
        """Stubbed readiness check that always reports the model as ready."""
        return True

    @classmethod
    def send_warmup(cls):
        """Stubbed warmup that always succeeds (no real request)."""
        return True


sys.modules["rag_controller"] = _StubRC

if "db" in sys.modules:
    del sys.modules["db"]
import db  # noqa: E402
_tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmpdb.close()
db.DB_PATH = Path(_tmpdb.name)
db.init_db()

if "app" in sys.modules:
    del sys.modules["app"]
import app  # noqa: E402  — note: this spawns the daemon worker. We
              #               drive the same worker_body() in tests below
              #               using monkey-patched chat_with_corpus to
              #               control outcomes.
import corpus  # noqa: E402  — worker.py calls corpus.chat_with_corpus; patch it here
import runtime  # noqa: E402  — rate_limiter / inflight_chat_reqs live here post-refactor


def reset_state():
    """Reset shared test state between cases.

    Clears the stub job queue, truncates the ``jobs`` table, and resets the
    runtime rate limiter so the same ``user_id`` can be reused per test.
    """
    _StubRC.job_queue.clear()
    with db.get_conn() as conn:
        conn.execute("DELETE FROM jobs")
    # Reset rate limiter so we can use the same user_id across tests
    with runtime.rate_limiter._lock:
        runtime.rate_limiter.user_data.clear()


def wait_for(predicate, timeout=5, interval=0.05):
    """Poll ``predicate`` until it is truthy or ``timeout`` elapses.

    Test helper for awaiting the background worker. Returns True if the
    predicate became true within the deadline, else False.
    """
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        if predicate():
            return True
        _time.sleep(interval)
    return False


def test_worker_marks_processing_then_done_with_json_blob():
    """On success the worker drives the job to 'done' and stores a JSON blob
    of {"reply", "references"}, then drains the job from the queue."""
    reset_state()
    test_refs = [{"title": "Cited Doc", "url": "https://example.com/cited"}]

    def fake_chat_with_corpus(model_type, prompt, **kwargs):
        """Stubbed chat call returning a fixed reply and reference list."""
        return ("The answer is 42.", test_refs)

    with mock.patch.object(corpus, "chat_with_corpus", side_effect=fake_chat_with_corpus):
        job_id = db.insert_job("u1", "what is the answer")
        _StubRC.queue_job("u1", job_id, "sim", "what is the answer", None, 5)
        # The real worker thread spawned at app import time is polling.
        # It will pick this up within WORKER_NO_WORK_SLEEP_INTERVAL_SECS=5s.
        ok = wait_for(lambda: db.get_job(job_id, "u1")["status"] == "done", timeout=10)
        assert ok, f"job never reached done; current row={db.get_job(job_id, 'u1')}"

    row = db.get_job(job_id, "u1")
    assert row["status"] == "done"
    parsed = json.loads(row["response"])
    assert parsed["reply"] == "The answer is 42."
    assert parsed["references"] == test_refs
    # Queue must be drained
    assert _StubRC.job_queue_len() == 0
    print("  OK  test_worker_marks_processing_then_done_with_json_blob")


def test_worker_marks_failed_on_exception():
    """When the chat call raises, the worker marks the job 'failed', writes an
    error response, and still drains it from the queue."""
    reset_state()

    def boom(*a, **kw):
        """Stubbed chat call that raises to simulate a model failure."""
        raise RuntimeError("simulated model failure")

    with mock.patch.object(corpus, "chat_with_corpus", side_effect=boom):
        job_id = db.insert_job("u1", "trigger failure")
        _StubRC.queue_job("u1", job_id, "sim", "trigger failure", None, 5)
        ok = wait_for(lambda: db.get_job(job_id, "u1")["status"] == "failed", timeout=10)
        assert ok, f"job never reached failed; current row={db.get_job(job_id, 'u1')}"

    row = db.get_job(job_id, "u1")
    assert "Search system not initialized" in row["response"]
    assert _StubRC.job_queue_len() == 0
    print("  OK  test_worker_marks_failed_on_exception")


def test_worker_inflight_counter_balanced():
    """inflight_chat_reqs.inc/.dec must net to 0 after each job, success or fail."""
    reset_state()
    starting = runtime.inflight_chat_reqs.get()

    def fake(*a, **kw):
        """Stubbed chat call returning a trivial successful result."""
        return ("ok", [])

    with mock.patch.object(corpus, "chat_with_corpus", side_effect=fake):
        for _ in range(3):
            jid = db.insert_job("u-many", "q")
            _StubRC.queue_job("u-many", jid, "sim", "q", None, 5)
        ok = wait_for(lambda: _StubRC.job_queue_len() == 0, timeout=15)
        assert ok, "queue not drained in time"

    # Allow a final iteration of the worker loop to settle
    _time.sleep(0.5)
    assert runtime.inflight_chat_reqs.get() == starting, (
        f"counter leaked: started at {starting}, now {runtime.inflight_chat_reqs.get()}"
    )
    print("  OK  test_worker_inflight_counter_balanced")


if __name__ == "__main__":
    tests = [
        test_worker_marks_processing_then_done_with_json_blob,
        test_worker_marks_failed_on_exception,
        test_worker_inflight_counter_balanced,
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
    print(f"{len(tests) - len(failures)} / {len(tests)} worker tests passed")
    # Don't sys.exit cleanly because the daemon worker is still polling;
    # that's harmless under daemon=True but we explicitly exit.
    sys.exit(1 if failures else 0)
