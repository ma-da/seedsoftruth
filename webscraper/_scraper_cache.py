"""
SQLite-backed cache for the web scraper.

Two tables:

    downloads    URL -> on-disk file metadata (path, size, hash, content type).
                 Used to skip re-downloading content that's already in the
                 corpus and to validate that the file on disk still matches.

    url_queue    URL -> (depth_actual, depth_effective). Persisted pending
                 work. On startup a crawl can re-enqueue any URLs that were
                 in flight when a previous run was interrupted.

Each connection is opened with ``journal_mode=WAL`` and ``busy_timeout=5000``
so that the multi-threaded crawler doesn't trip "database is locked" under
contention.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from queue import Queue
from typing import Optional, Tuple

log = logging.getLogger(__name__)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(db_path: str) -> None:
    """Create the two tables if they don't already exist. Idempotent."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS downloads (
                cleaned_url    TEXT PRIMARY KEY,
                content_type   TEXT NOT NULL,
                url_file_path  TEXT NOT NULL,
                url_file_size  INTEGER NOT NULL,
                text_file_path TEXT NOT NULL,
                text_file_size INTEGER NOT NULL,
                hash           TEXT NOT NULL,
                download_time  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS url_queue (
                url             TEXT PRIMARY KEY,
                depth_actual    INTEGER NOT NULL,
                depth_effective INTEGER NOT NULL
            )
        """)
        conn.commit()
    log.debug("cache db initialized at %s", db_path)


def table_exists(db_path: str, table_name: str) -> bool:
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            return row is not None
    except sqlite3.Error as e:
        log.error("cache db error checking table %s: %s", table_name, e)
        return False


# --------------------------------------------------------------------------- #
# downloads table                                                             #
# --------------------------------------------------------------------------- #

def update_cache(db_path: str, cleaned_url: str, content_type: str,
                 url_file_path: str, url_file_size: int,
                 text_file_path: str, text_file_size: int,
                 content_hash: str,
                 download_time: Optional[datetime] = None) -> None:
    """Insert or replace the cache entry for `cleaned_url`."""
    if download_time is None:
        download_time = datetime.now()
    with _connect(db_path) as conn:
        conn.execute("""
            INSERT INTO downloads (
                cleaned_url, content_type, url_file_path, url_file_size,
                text_file_path, text_file_size, hash, download_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cleaned_url) DO UPDATE SET
                url_file_path  = excluded.url_file_path,
                url_file_size  = excluded.url_file_size,
                text_file_path = excluded.text_file_path,
                text_file_size = excluded.text_file_size,
                download_time  = excluded.download_time,
                hash           = excluded.hash
        """, (cleaned_url, content_type, url_file_path, url_file_size,
              text_file_path, text_file_size, content_hash,
              download_time.isoformat()))
        conn.commit()


def remove_download_entry(db_path: str, cleaned_url: str) -> None:
    try:
        with _connect(db_path) as conn:
            conn.execute("DELETE FROM downloads WHERE cleaned_url = ?",
                         (cleaned_url,))
            conn.commit()
    except sqlite3.Error as e:
        log.error("error removing entry from downloads: %s", e)


def get_cached_file_content(db_path: str, cleaned_url: str
                            ) -> Optional[Tuple[bytes, str]]:
    """Return (content_bytes, content_type) if the cached file still matches.

    The cache row is removed if the file is missing, since it is no longer
    truthful. Returns None if there is no usable cached copy.
    """
    with _connect(db_path) as conn:
        row = conn.execute("""
            SELECT url_file_path, url_file_size, content_type
            FROM downloads WHERE cleaned_url = ?
        """, (cleaned_url,)).fetchone()

    if not row:
        return None

    file_path, expected_size, content_type = row
    if not os.path.exists(file_path):
        log.debug("cache: removing stale entry for %s (file missing)", cleaned_url)
        remove_download_entry(db_path, cleaned_url)
        return None

    if os.path.getsize(file_path) != expected_size:
        log.debug("cache: size mismatch for %s; ignoring entry", cleaned_url)
        return None

    with open(file_path, "rb") as f:
        return f.read(), content_type


def clear_downloads(db_path: str, delete_db: bool = False) -> None:
    """Wipe the downloads table, or delete the entire DB file."""
    if delete_db and os.path.exists(db_path):
        os.remove(db_path)
        log.debug("deleted cache database: %s", db_path)
        return
    if not os.path.exists(db_path):
        return
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM downloads")
        conn.commit()
    log.debug("cleared downloads table at %s", db_path)


# --------------------------------------------------------------------------- #
# pending url queue                                                           #
# --------------------------------------------------------------------------- #

def save_pending_url(db_path: str, url: str,
                     depth_actual: int, depth_effective: int) -> None:
    with _connect(db_path) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO url_queue (url, depth_actual, depth_effective)
            VALUES (?, ?, ?)
        """, (url, depth_actual, depth_effective))
        conn.commit()


def delete_pending_url(db_path: str, url: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM url_queue WHERE url = ?", (url,))
        conn.commit()


def load_pending_urls(db_path: str, url_queue: Queue) -> int:
    """Push every persisted pending URL onto `url_queue`. Returns count added."""
    initial = url_queue.qsize()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT url, depth_actual, depth_effective FROM url_queue"
        ).fetchall()
    for url, depth_actual, depth_effective in rows:
        url_queue.put((url, depth_actual, depth_effective))
    added = url_queue.qsize() - initial
    log.info("loaded %d pending URLs from %s", added, db_path)
    return added


def clear_pending_queue(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM url_queue")
        conn.commit()
