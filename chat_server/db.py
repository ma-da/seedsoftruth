"""SQLite persistence layer for the Seeds of Truth app.

This module owns two responsibilities:

1. The application database (``db/app.db``), which holds two tables:

   - ``jobs``: one row per generation request, keyed by a UUID ``id``.
     Tracks the owning ``user_id``, the ``prompt``, a ``status`` of
     ``queued``/``processing``/``done``/``failed``, ``created_at`` and
     ``completed_at`` timestamps, and the generated ``response``. Two
     v2 columns support optional email delivery of the response:
     ``email`` (the destination address) and ``email_status``
     (``NULL``/``pending``/``sending``/``sent``/``failed``).
   - ``feedback``: one row per user rating of a job, with integer
     ``relevance``, ``accuracy`` and ``style`` scores, optional free
     text ``comments``, and a ``created_at`` timestamp.

   ``init_db`` creates these tables and their indexes, enables WAL
   mode and foreign keys, and migrates older databases by adding the
   v2 email columns when they are absent.

2. Read-only lookups against the hybrid RAG database (whose path comes
   from ``rag_controller.HYBRID_DB_PATH``), exposing the entity and
   topic metadata associated with a content chunk.

Note: for the MVP the ``user_id`` is simply the caller's UUID; this
may change once dedicated user accounts exist.
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import rag_controller

DB_PATH = Path("db/app.db")


def init_db() -> None:
    """Create the application database, its schema, and its indexes.

    Ensures the parent directory of ``DB_PATH`` exists, opens a
    connection in autocommit mode, enables WAL journaling and foreign
    keys, and creates the ``jobs`` and ``feedback`` tables (and their
    indexes) if they do not already exist. Also migrates pre-v2
    databases by adding the ``email`` and ``email_status`` columns to
    ``jobs`` when missing. Intended to be called once at startup.

    Raises:
        PermissionError: If the database directory cannot be created.
        sqlite3.Error: If schema initialization fails; treated as a
            fatal startup error.
    """
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logging.critical(
            f"No permission to create DB directory: {DB_PATH.parent}"
        )
        raise

    logging.info(f"Using database at: {DB_PATH.resolve()}")

    # Use a generous timeout to tolerate concurrent startup
    conn = sqlite3.connect(
        DB_PATH, timeout=30, isolation_level=None  # autocommit mode
    )

    try:
        # Harden SQLite behavior
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=30000;")

        # Schema initialization (atomic)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            status TEXT CHECK(status IN ('queued','processing','done','failed')) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            completed_at TIMESTAMP,
            response TEXT,
            email TEXT,
            email_status TEXT
        );
        """)

        # Migration: add email columns to pre-existing jobs tables that were
        # created before the email-response feature landed. PRAGMA reads cheaply
        # and ALTER is idempotent-guarded by the column check.
        # Note: the init_db() connection doesn't set row_factory=sqlite3.Row
        # (only get_conn() does), so PRAGMA rows come back as plain tuples.
        # The column at index 1 of PRAGMA table_info is the column name.
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(jobs)")
        }
        if "email" not in existing_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN email TEXT")
            logging.info("Added 'email' column to jobs table")
        if "email_status" not in existing_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN email_status TEXT")
            logging.info("Added 'email_status' column to jobs table")

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created
        ON jobs(status, created_at);
        """)

        # Index on email_status helps the worker quickly find jobs that still
        # need an email sent. Most rows have email_status NULL so a partial
        # index would be tighter, but SQLite supports it and the predicate is
        # simple enough.
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_email_status
        ON jobs(email_status)
        WHERE email_status IS NOT NULL;
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_user_created
        ON jobs(user_id, created_at);
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            job_id TEXT NOT NULL,
            relevance INTEGER NOT NULL,
            accuracy INTEGER NOT NULL,
            style INTEGER NOT NULL,
            comments TEXT NULL,
            created_at TIMESTAMP NOT NULL
        );
        """)

        conn.execute("""CREATE INDEX IF NOT EXISTS idx_feedback_job_id
        ON feedback (job_id);
        """)

        logging.info("Database initialized successfully")

    except sqlite3.Error as e:
        logging.exception("Database initialization failed")
        raise  # fail fast — this is a fatal startup error

    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    """Open a connection to the application database.

    The connection uses autocommit mode, a 30-second busy timeout,
    allows cross-thread use, and sets ``row_factory`` to
    ``sqlite3.Row`` so query results support mapping access.

    Returns:
        sqlite3.Connection: A new connection to ``DB_PATH``.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def get_hybrid_db_conn() -> sqlite3.Connection:
    """Open a connection to the hybrid RAG database.

    Connects to ``rag_controller.HYBRID_DB_PATH`` with the same
    settings as :func:`get_conn` (autocommit, cross-thread,
    ``sqlite3.Row`` factory). Used for read-only entity and topic
    lookups.

    Returns:
        sqlite3.Connection: A new connection to the hybrid database.
    """
    conn = sqlite3.connect(
        rag_controller.HYBRID_DB_PATH,
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def insert_job(user_id: str, prompt: str) -> str:
    """Insert a new job in the ``queued`` state.

    Generates a fresh UUID for the job id and records the current UTC
    time as ``created_at``.

    Args:
        user_id: Identifier of the user who owns the job.
        prompt: The generation prompt for the job.

    Returns:
        str: The newly generated job id.
    """
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
        INSERT INTO jobs (id, user_id, prompt, status, created_at)
        VALUES (?, ?, ?, 'queued', ?)
        """,
            (job_id, user_id, prompt, now),
        )

    return job_id


def get_job(job_id: str, user_id: str) -> Optional[Dict]:
    """Fetch a single job owned by the given user.

    Args:
        job_id: The id of the job to retrieve.
        user_id: The id of the user the job must belong to.

    Returns:
        Optional[Dict]: A dict of the full ``jobs`` row keyed by
        column name, or ``None`` if no matching job exists.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
        SELECT * FROM jobs
        WHERE id = ? AND user_id = ?
        """,
            (job_id, user_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_queued_jobs(limit: int = 10) -> List[Dict]:
    """Return the oldest queued jobs awaiting processing.

    Args:
        limit: Maximum number of jobs to return.

    Returns:
        List[Dict]: A list of ``jobs`` rows (each a dict keyed by
        column name) with ``status = 'queued'``, ordered by
        ``created_at`` ascending (oldest first).
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
        SELECT * FROM jobs
        WHERE status = 'queued'
        ORDER BY created_at ASC
        LIMIT ?
        """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_jobs_for_user(user_id: str, limit: int = 20) -> List[Dict]:
    """Return a user's most recent jobs.

    Args:
        user_id: The id of the user whose jobs to retrieve.
        limit: Maximum number of jobs to return.

    Returns:
        List[Dict]: A list of ``jobs`` rows (each a dict keyed by
        column name) owned by the user, ordered by ``created_at``
        descending (newest first).
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
        SELECT * FROM jobs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
            (user_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def mark_processing(job_id: str) -> bool:
    """Atomically claim a queued job for processing.

    Transitions the job from ``queued`` to ``processing`` only if it
    is currently ``queued``, so concurrent workers cannot both claim
    the same job.

    Args:
        job_id: The id of the job to claim.

    Returns:
        bool: ``True`` if this caller claimed the job (the row was
        updated), ``False`` otherwise.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
        UPDATE jobs
        SET status = 'processing'
        WHERE id = ? AND status = 'queued'
        """,
            (job_id,),
        )
        return cur.rowcount == 1


def mark_done(job_id: str, response: str) -> None:
    """Mark a job as completed and store its generated response.

    Sets ``status = 'done'``, records the current UTC time as
    ``completed_at``, and writes the response text.

    Args:
        job_id: The id of the job to update.
        response: The generated response text to store.
    """
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
        UPDATE jobs
        SET status = 'done',
            completed_at = ?,
            response = ?
        WHERE id = ?
        """,
            (now, response, job_id),
        )


def mark_failed(job_id: str, error_msg: str) -> None:
    """Mark a job as failed and store the error message.

    Sets ``status = 'failed'``, records the current UTC time as
    ``completed_at``, and stores the error message in the
    ``response`` column.

    Args:
        job_id: The id of the job to update.
        error_msg: A description of the failure, stored in
            ``response``.
    """
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
        UPDATE jobs
        SET status = 'failed',
            completed_at = ?,
            response = ?
        WHERE id = ?
        """,
            (now, error_msg, job_id),
        )


def sweep_orphaned_jobs(stale_minutes: int = 10) -> int:
    """
    Mark any 'queued' or 'processing' jobs older than `stale_minutes` as
    'failed'. Intended to be called once at app startup so a gunicorn restart
    that loses the in-process queue doesn't leave zombie rows that the client
    polls forever.

    The threshold (default 10 min) is well above the worst-case real LLM
    latency (~3 min) but well below typical session timeouts, so a healthy
    request will never get swept while in flight, and a restart-orphan will
    get cleaned up within one polling window of the next boot.

    Returns the number of rows swept.
    """
    from datetime import timedelta

    cutoff = (datetime.utcnow() - timedelta(minutes=stale_minutes)).isoformat()
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        cur = conn.execute("""
        UPDATE jobs
        SET status = 'failed',
            completed_at = ?,
            response = ?
        WHERE status IN ('queued', 'processing')
          AND created_at < ?
        """, (now, "server restarted before job completed", cutoff))
        swept = cur.rowcount

    if swept:
        logging.warning(
            "sweep_orphaned_jobs: marked %d stale job(s) as failed "
            "(threshold=%d min, cutoff=%s)",
            swept, stale_minutes, cutoff,
        )
    else:
        logging.info(
            "sweep_orphaned_jobs: no stale jobs found (threshold=%d min)",
            stale_minutes,
        )
    return swept


def attach_email(job_id: str, email: str) -> str:
    """
    Attach an email address to a job for later delivery of the response.
    Sets email_status='pending' only when it's safe to do so — i.e. the
    row exists and the email hasn't already been sent (or is mid-send).

    Returns one of:
      'attached'      — row was found and email_status now = 'pending'
      'already_sent'  — row was found but email is already 'sent' or
                        'sending' (a duplicate POST after the worker /
                        race-branch already started/finished delivery)
      'not_found'     — no job with that id

    The conditional UPDATE makes this idempotent against the worker
    racing the API's race-branch: the worker can only claim a row
    where email_status='pending', so attempting to overwrite 'sent'
    or 'sending' is rejected here and the duplicate POST short-circuits.
    """
    with get_conn() as conn:
        # Single atomic UPDATE — only writes if the row is in a state
        # where re-attaching makes sense (never overwrite 'sent' or
        # 'sending').
        cur = conn.execute(
            """
            UPDATE jobs
            SET email = ?,
                email_status = 'pending'
            WHERE id = ?
              AND (email_status IS NULL
                   OR email_status IN ('pending', 'failed'))
            """,
            (email, job_id),
        )
        if cur.rowcount == 1:
            return "attached"

        # No update happened — figure out whether it's missing or
        # locked in a terminal/in-flight state.
        check = conn.execute(
            "SELECT email_status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if check is None:
            return "not_found"
        status = check["email_status"]
        if status in ("sent", "sending"):
            return "already_sent"
        # Unexpected: row exists in a state we don't recognize. Treat
        # as not_found so callers don't silently double-send.
        return "not_found"


def claim_email_for_send(job_id: str) -> bool:
    """
    Atomic 'pending → sending' transition. Returns True iff the
    caller now owns the send (i.e. nobody else has already claimed
    it).

    This is the synchronization primitive that prevents the worker's
    post-mark_done hook and the API's race-branch from BOTH sending
    when a user POSTs /api/email_response in the microsecond window
    between db.mark_done() and _maybe_send_response_email(). Only one
    of them will see rowcount==1; the other no-ops.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE jobs
            SET email_status = 'sending'
            WHERE id = ?
              AND email_status = 'pending'
            """,
            (job_id,),
        )
        return cur.rowcount == 1


def get_email_for_job(job_id: str) -> Optional[Dict]:
    """
    Returns a dict with the job's email-related fields, or None if the job
    doesn't exist. Useful for the worker after mark_done / mark_failed.

    Shape: {"email": str|None, "email_status": str|None,
            "status": str, "response": str|None, "prompt": str|None}
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
        SELECT email, email_status, status, response, prompt
        FROM jobs
        WHERE id = ?
        """,
            (job_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def mark_email_status(job_id: str, status: str) -> bool:
    """
    Update the email delivery status for a job.

    Allowed values: 'pending', 'sending', 'sent', 'failed'. None is also
    accepted to clear the marker (e.g. user cancels their request
    before send). 'sending' is the in-flight state set by
    claim_email_for_send and cleared by the eventual 'sent' / 'failed'
    transition.
    """
    if status is not None and status not in (
        "pending",
        "sending",
        "sent",
        "failed",
    ):
        raise ValueError(f"invalid email_status: {status!r}")

    with get_conn() as conn:
        cur = conn.execute(
            """
        UPDATE jobs
        SET email_status = ?
        WHERE id = ?
        """,
            (status, job_id),
        )
        return cur.rowcount == 1


def clear_email(job_id: str) -> bool:
    """
    Clears the email and email_status. Used when the user cancels an
    email request before the job completes.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """
        UPDATE jobs
        SET email = NULL,
            email_status = NULL
        WHERE id = ?
        """,
            (job_id,),
        )
        return cur.rowcount == 1


def job_exists(job_id: str) -> bool:
    """Return whether a job with the given id exists.

    Args:
        job_id: The id to look up in the ``jobs`` table.

    Returns:
        bool: ``True`` if a row with that id exists in ``jobs``,
        ``False`` otherwise.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1
            FROM jobs
            WHERE id = ?
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone() is not None


def feedback_exists_for_job(job_id: str) -> bool:
    """Return whether any feedback row exists for the given job.

    Args:
        job_id: The job id to check in the ``feedback`` table.

    Returns:
        bool: ``True`` if at least one ``feedback`` row references the
        job id, ``False`` otherwise.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1
            FROM feedback
            WHERE job_id = ?
            LIMIT 1
            """,
            (job_id,),
        )
        return cur.fetchone() is not None


def delete_job(job_id: str, user_id: str) -> None:
    """Delete a job owned by the given user.

    The job is only removed if both the id and the owning ``user_id``
    match; deleting a non-existent or non-owned job is a no-op.

    Args:
        job_id: The id of the job to delete.
        user_id: The id of the user the job must belong to.
    """
    with get_conn() as conn:
        conn.execute(
            """
        DELETE FROM jobs
        WHERE id = ? AND user_id = ?
        """,
            (job_id, user_id),
        )


def insert_feedback(
    job_id: str,
    relevance: int,
    accuracy: int,
    style: int,
    comments: Optional[str] = None,
) -> int:
    """Insert a feedback row for a job and return its id.

    Records the current UTC time as ``created_at``.

    Args:
        job_id: The id of the job the feedback is about.
        relevance: Integer relevance rating.
        accuracy: Integer accuracy rating.
        style: Integer style rating.
        comments: Optional free-text comments.

    Returns:
        int: The auto-generated ``feedback.id`` of the new row.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (
                job_id,
                relevance,
                accuracy,
                style,
                comments,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                relevance,
                accuracy,
                style,
                comments,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def delete_feedback(conn: sqlite3.Connection, feedback_id: int) -> bool:
    """Delete a feedback row by its id.

    Args:
        conn: An open connection to the application database.
        feedback_id: The id of the ``feedback`` row to delete.

    Returns:
        bool: ``True`` if a row was deleted, ``False`` if no row
        matched.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
    conn.commit()
    return cur.rowcount > 0


def get_feedback_for_job(conn: sqlite3.Connection, job_id: str) -> List[Dict]:
    """Return all feedback entries for a job, newest first.

    Args:
        conn: An open connection to the application database.
        job_id: The id of the job whose feedback to retrieve.

    Returns:
        List[Dict]: A list of ``feedback`` rows (each a dict keyed by
        column name), ordered by ``created_at`` descending.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM feedback
        WHERE job_id = ?
        ORDER BY created_at DESC
        """,
        (job_id,),
    )
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def get_feedback_by_id(
    conn: sqlite3.Connection, feedback_id: int
) -> Optional[Dict]:
    """Return a single feedback row by its id.

    Args:
        conn: An open connection to the application database.
        feedback_id: The id of the ``feedback`` row to retrieve.

    Returns:
        Optional[Dict]: The ``feedback`` row as a dict keyed by column
        name, or ``None`` if no row matched.
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_chunk_entities(chunk_id: str) -> list[dict]:
    """Return the entity metadata associated with a content chunk.

    Queries the hybrid RAG database, joining ``chunks`` to
    ``entities`` via ``chunk_entities``, ordered by entity type then
    canonical name.

    Args:
        chunk_id: The public chunk id to look up.

    Returns:
        list[dict]: One dict per linked entity, each with keys
        ``"name"`` (the entity's canonical name) and ``"type"`` (the
        entity type, e.g. ``"person"``). Empty if the chunk has no
        linked entities.
    """
    with get_hybrid_db_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                e.canonical_name,
                e.type
            FROM chunks c
            JOIN chunk_entities ce
              ON c.lookup_id = ce.chunk_lookup_id
            JOIN entities e
              ON ce.entity_id = e.entity_id
            WHERE c.chunk_id = ?
            ORDER BY e.type, e.canonical_name
            """,
            (chunk_id,),
        )

        rows = cur.fetchall()

        return [
            {
                "name": row["canonical_name"],
                "type": row["type"],
            }
            for row in rows
        ]


def get_chunk_topics(chunk_id: str) -> list[dict]:
    """Return the topic metadata associated with a content chunk.

    Queries the hybrid RAG database, joining ``chunks`` to ``topics``
    via ``chunk_topics``, ordered by domain then topic name.

    Args:
        chunk_id: The public chunk id to look up.

    Returns:
        list[dict]: One dict per linked topic, each with keys
        ``"domain"`` (the topic's domain) and ``"topic"`` (the topic
        name). Empty if the chunk has no linked topics.
    """
    with get_hybrid_db_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                t.domain,
                t.topic_name
            FROM chunks c
            JOIN chunk_topics ct
              ON c.lookup_id = ct.chunk_lookup_id
            JOIN topics t
              ON ct.topic_id = t.topic_id
            WHERE c.chunk_id = ?
            ORDER BY t.domain, t.topic_name
            """,
            (chunk_id,),
        )

        rows = cur.fetchall()

        return [
            {
                "domain": row["domain"],
                "topic": row["topic_name"],
            }
            for row in rows
        ]
