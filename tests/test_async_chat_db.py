"""
Direct unit tests for the db.py changes from the async-chat migration.

Tested:
  - sweep_orphaned_jobs marks stale queued/processing rows as failed,
    leaves fresh rows alone, and is a no-op when nothing is stale.
  - mark_processing CAS is atomic (queued→processing only).
  - JSON-blob round trip via mark_done + get_job, including legacy
    plain-string compatibility.

Run from anywhere: python3 test_async_chat_db.py
"""
import importlib
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

# db.py does `import rag_controller` at module load just to read
# HYBRID_DB_PATH from one function we don't exercise. Stub the import so
# we don't drag in spacy/bm25s/etc. just to test schema + CRUD.
sys.modules.setdefault(
    "rag_controller",
    mock.MagicMock(HYBRID_DB_PATH="/tmp/_unused_hybrid.db"),
)

REPO = str(Path(__file__).resolve().parent.parent)
# Web-server modules moved under chat_server/ but import each other flat.
sys.path.insert(0, str(Path(REPO) / "chat_server"))

# Re-import each run to pick up code changes
if "db" in sys.modules:
    del sys.modules["db"]
import db


def setup_isolated_db():
    """Point db.DB_PATH at a fresh tempfile so each test is hermetic."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = Path(tmp.name)
    db.init_db()
    return tmp.name


def insert_job_with_age(user_id, prompt, age_minutes, status="queued"):
    """Insert a row directly so we can control created_at."""
    import uuid as _uuid
    job_id = str(_uuid.uuid4())
    created_at = (datetime.utcnow() - timedelta(minutes=age_minutes)).isoformat()
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (id, user_id, prompt, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, user_id, prompt, status, created_at),
        )
    return job_id


def test_sweep_marks_stale_queued_as_failed():
    """A 'queued' job older than stale_minutes is swept to 'failed' (with a
    'server restarted' note) while a fresh queued job is left untouched."""
    setup_isolated_db()
    fresh = db.insert_job("u1", "fresh question")        # created_at = now
    stale = insert_job_with_age("u1", "stale q", 30)     # 30 min old, queued

    swept = db.sweep_orphaned_jobs(stale_minutes=10)
    assert swept == 1, f"expected 1 sweep, got {swept}"

    fresh_row = db.get_job(fresh, "u1")
    stale_row = db.get_job(stale, "u1")
    assert fresh_row["status"] == "queued", f"fresh row was clobbered: {fresh_row}"
    assert stale_row["status"] == "failed", f"stale row not swept: {stale_row}"
    assert "server restarted" in stale_row["response"]
    print("  OK  test_sweep_marks_stale_queued_as_failed")


def test_sweep_marks_stale_processing_as_failed():
    """A 'processing' job stuck past stale_minutes is swept to 'failed'."""
    setup_isolated_db()
    stale = insert_job_with_age("u1", "stuck mid-call", 20, status="processing")
    swept = db.sweep_orphaned_jobs(stale_minutes=10)
    assert swept == 1
    row = db.get_job(stale, "u1")
    assert row["status"] == "failed"
    print("  OK  test_sweep_marks_stale_processing_as_failed")


def test_sweep_leaves_done_and_failed_alone():
    """The sweep ignores terminal rows: old 'done' and 'failed' jobs keep
    their status and are not counted as swept."""
    setup_isolated_db()
    done = insert_job_with_age("u1", "done q", 60)
    db.mark_done(done, "ok")
    failed = insert_job_with_age("u1", "failed q", 60)
    db.mark_failed(failed, "boom")

    swept = db.sweep_orphaned_jobs(stale_minutes=10)
    assert swept == 0, f"sweep should ignore terminal rows, got {swept}"

    assert db.get_job(done, "u1")["status"] == "done"
    assert db.get_job(failed, "u1")["status"] == "failed"
    print("  OK  test_sweep_leaves_done_and_failed_alone")


def test_sweep_noop_when_nothing_stale():
    """The sweep is a no-op (returns 0) when all queued jobs are fresh."""
    setup_isolated_db()
    db.insert_job("u1", "fresh1")
    db.insert_job("u1", "fresh2")
    swept = db.sweep_orphaned_jobs(stale_minutes=10)
    assert swept == 0
    print("  OK  test_sweep_noop_when_nothing_stale")


def test_mark_processing_cas_only_transitions_from_queued():
    """mark_processing is an atomic compare-and-swap: it succeeds once from
    'queued' and returns False on a row already in 'processing'."""
    setup_isolated_db()
    job_id = db.insert_job("u1", "question")
    assert db.mark_processing(job_id) is True, "first transition should succeed"
    # Second call on a row already in 'processing' must NOT match the
    # CAS — that would let two workers grab the same job in a multi-worker
    # future.
    assert db.mark_processing(job_id) is False, "second call should fail (already processing)"
    print("  OK  test_mark_processing_cas_only_transitions_from_queued")


def test_json_blob_round_trip():
    """Worker writes a JSON blob; /api/job/<id> reads it back. Verify the
    blob shape used by api_job_status survives mark_done."""
    setup_isolated_db()
    job_id = db.insert_job("u1", "question")
    db.mark_processing(job_id)

    payload = json.dumps({
        "reply": "Hello world",
        "references": [
            {"title": "Doc A", "url": "https://example.com/a"},
            {"title": "Doc B", "url": "https://example.com/b"},
        ],
    })
    db.mark_done(job_id, payload)

    row = db.get_job(job_id, "u1")
    assert row["status"] == "done"
    assert row["response"] == payload
    parsed = json.loads(row["response"])
    assert parsed["reply"] == "Hello world"
    assert len(parsed["references"]) == 2
    print("  OK  test_json_blob_round_trip")


def test_legacy_plain_string_response_still_readable():
    """Pre-migration rows have plain-string answers in the response column.
    The api_job_status fallback should still surface them. We can't test
    the route here, but we verify the row layout that the route expects."""
    setup_isolated_db()
    job_id = db.insert_job("u1", "old question")
    db.mark_done(job_id, "Just a string answer with no JSON")
    row = db.get_job(job_id, "u1")
    assert row["response"] == "Just a string answer with no JSON"
    # The route's heuristic: doesn't start with '{', so treat as legacy reply.
    assert not row["response"].lstrip().startswith("{")
    print("  OK  test_legacy_plain_string_response_still_readable")


def test_get_job_returns_none_for_wrong_user():
    """get_job is scoped to the owner: it returns None for a mismatched
    user_id or an unknown job_id, preventing cross-user leakage."""
    setup_isolated_db()
    job_id = db.insert_job("u1", "question")
    assert db.get_job(job_id, "u1") is not None
    assert db.get_job(job_id, "u2") is None, "must not leak to other users"
    assert db.get_job("does-not-exist", "u1") is None
    print("  OK  test_get_job_returns_none_for_wrong_user")


if __name__ == "__main__":
    tests = [
        test_sweep_marks_stale_queued_as_failed,
        test_sweep_marks_stale_processing_as_failed,
        test_sweep_leaves_done_and_failed_alone,
        test_sweep_noop_when_nothing_stale,
        test_mark_processing_cas_only_transitions_from_queued,
        test_json_blob_round_trip,
        test_legacy_plain_string_response_still_readable,
        test_get_job_returns_none_for_wrong_user,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}: {e}")
    print()
    print(f"{len(tests) - len(failures)} / {len(tests)} db tests passed")
    sys.exit(1 if failures else 0)
