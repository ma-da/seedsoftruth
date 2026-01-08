
"""
Schemas we will use for our sqlite db

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT CHECK(status IN ('queued','processing','done','failed')) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    response TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    job_id TEXT NOT NULL,
    relevance INTEGER NOT NULL,
    accuracy INTEGER NOT NULL,
    style INTEGER NOT NULL,
    comments TEXT NULL,
    created_at TIMESTAMP NOT NULL
);

- For MVP, we will use the uuid as the user_id. This might change when we have dedicated user accounts.
);
"""
import logging
import sqlite3
import uuid
from datetime import datetime

from pathlib import Path
from typing import Optional, List, Dict

DB_PATH = Path("db/app.db")

def init_db():
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logging.critical(f"No permission to create DB directory: {DB_PATH.parent}")
        raise

    logging.info(f"Using database at: {DB_PATH.resolve()}")

    # Use a generous timeout to tolerate concurrent startup
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None  # autocommit mode
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
            response TEXT
        );
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created
        ON jobs(status, created_at);
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

def get_conn():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def insert_job(user_id: str, prompt: str) -> str:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute("""
        INSERT INTO jobs (id, user_id, prompt, status, created_at)
        VALUES (?, ?, ?, 'queued', ?)
        """, (job_id, user_id, prompt, now))

    return job_id


def get_job(job_id: str, user_id: str):
    with get_conn() as conn:
        cur = conn.execute("""
        SELECT * FROM jobs
        WHERE id = ? AND user_id = ?
        """, (job_id, user_id))
        row = cur.fetchone()
        return dict(row) if row else None


def get_queued_jobs(limit=10):
    with get_conn() as conn:
        cur = conn.execute("""
        SELECT * FROM jobs
        WHERE status = 'queued'
        ORDER BY created_at ASC
        LIMIT ?
        """, (limit,))
        return [dict(r) for r in cur.fetchall()]


def get_jobs_for_user(user_id: str, limit=20):
    with get_conn() as conn:
        cur = conn.execute("""
        SELECT * FROM jobs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """, (user_id, limit))
        return [dict(r) for r in cur.fetchall()]


def mark_processing(job_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("""
        UPDATE jobs
        SET status = 'processing'
        WHERE id = ? AND status = 'queued'
        """, (job_id,))
        return cur.rowcount == 1


def mark_done(job_id: str, response: str):
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute("""
        UPDATE jobs
        SET status = 'done',
            completed_at = ?,
            response = ?
        WHERE id = ?
        """, (now, response, job_id))


def mark_failed(job_id: str, error_msg: str):
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        conn.execute("""
        UPDATE jobs
        SET status = 'failed',
            completed_at = ?,
            response = ?
        WHERE id = ?
        """, (now, error_msg, job_id))


def job_exists(
        job_id: str
) -> bool:
    """
    Returns True if at least one feedback row exists for the given job_id.
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
            (job_id,)
        )
        return cur.fetchone() is not None

def feedback_exists_for_job(
    job_id: str
) -> bool:
    """
    Returns True if at least one feedback row exists for the given job_id.
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
            (job_id,)
        )
        return cur.fetchone() is not None


def delete_job(job_id: str, user_id: str):
    with get_conn() as conn:
        conn.execute("""
        DELETE FROM jobs
        WHERE id = ? AND user_id = ?
        """, (job_id, user_id))

def insert_feedback(
    job_id: str,
    relevance: int,
    accuracy: int,
    style: int,
    comments: Optional[str] = None
) -> int:
    """
    Inserts a feedback row.
    Returns the newly created feedback.id
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
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        return cur.lastrowid

def delete_feedback(conn: sqlite3.Connection, feedback_id: int) -> bool:
    """
    Deletes a feedback row by id.
    Returns True if a row was deleted.
    """
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM feedback WHERE id = ?",
        (feedback_id,)
    )
    conn.commit()
    return cur.rowcount > 0

def get_feedback_for_job(
    conn: sqlite3.Connection,
    job_id: str
) -> List[Dict]:
    """
    Returns all feedback entries for a job_id,
    newest first.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM feedback
        WHERE job_id = ?
        ORDER BY created_at DESC
        """,
        (job_id,)
    )
    rows = cur.fetchall()
    return [dict(row) for row in rows]

def get_feedback_by_id(
    conn: sqlite3.Connection,
    feedback_id: int
) -> Optional[Dict]:
    """
    Returns a single feedback row or None.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM feedback WHERE id = ?",
        (feedback_id,)
    )
    row = cur.fetchone()
    return dict(row) if row else None
