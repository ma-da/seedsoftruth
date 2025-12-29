
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

- For MVP, we will use the uuid as the user_id. This might change when we have dedicated user accounts.
);
"""
import logging
import sqlite3
from pathlib import Path

DB_PATH = Path("db/app.db")

import sqlite3
import logging

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

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


def create_jobs_table():
    with get_conn() as conn:
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

def create_indexes():
    with get_conn() as conn:
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created
        ON jobs(status, created_at);
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_user_created
        ON jobs(user_id, created_at);
        """)


def create_jobs_table():
    with get_conn() as conn:
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

def create_indexes():
    with get_conn() as conn:
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created
        ON jobs(status, created_at);
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_user_created
        ON jobs(user_id, created_at);
        """)


import uuid
from datetime import datetime

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


def delete_job(job_id: str, user_id: str):
    with get_conn() as conn:
        conn.execute("""
        DELETE FROM jobs
        WHERE id = ? AND user_id = ?
        """, (job_id, user_id))
