#!/usr/bin/env python3
"""
Clear all entries from the Seeds of Truth application database.

This empties the ``jobs`` and ``feedback`` tables (and any other user
tables) in ``db/app.db`` while leaving the schema — tables, columns and
indexes — intact. Use it to reset local state before a check-in or a
fresh test run.

It does NOT touch the read-only hybrid RAG database
(``data/gamma_db_clean.db``); only the application DB is affected.

This is destructive and irreversible, so by default the script prints
the row counts it is about to delete and waits for you to type ``yes``.
Pass ``--yes`` to skip the prompt (e.g. in scripts or CI).

Typical usage:

    python scripts/clear_db.py                 # prompt, then wipe db/app.db
    python scripts/clear_db.py --yes           # no prompt
    python scripts/clear_db.py --db path/to.db # target a different DB

Stdlib-only; no requirements to install.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Default matches chat_server.db.DB_PATH; resolved relative to the repo root
# (the parent of this script's directory) so it works from any cwd.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "db" / "app.db"


def user_tables(conn: sqlite3.Connection) -> list[str]:
    """Return the names of user-defined tables, skipping SQLite internals."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def main() -> int:
    """Parse args, show what will be deleted, confirm, then clear the DB.

    Returns:
        Process exit code: 0 on success (or nothing to do), 1 if the DB
        file is missing, 3 if the user declines at the prompt.
    """
    p = argparse.ArgumentParser(
        description="Delete all rows from the application database "
        "(keeps the schema).",
    )
    p.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to the SQLite DB to clear (default: {DEFAULT_DB_PATH}).",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (non-interactive use).",
    )
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: database not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    try:
        tables = user_tables(conn)
        if not tables:
            print(f"No user tables in {db_path}; nothing to clear.")
            return 0

        counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                  for t in tables}
        total = sum(counts.values())

        print(f"Target database: {db_path.resolve()}")
        print("Rows to be deleted:")
        for t in tables:
            print(f"  {t}: {counts[t]}")
        print(f"  total: {total}")

        if total == 0:
            print("Database is already empty; nothing to do.")
            return 0

        if not args.yes:
            print()
            print("This permanently deletes all rows above. The schema is kept.")
            reply = input("Type 'yes' to continue: ").strip().lower()
            if reply != "yes":
                print("Aborted; no changes made.")
                return 3

        # Wipe inside a transaction, then reclaim disk space.
        conn.execute("PRAGMA foreign_keys=OFF;")
        with conn:
            for t in tables:
                conn.execute(f'DELETE FROM "{t}";')
            # Reset AUTOINCREMENT counters if the table exists.
            if conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='sqlite_sequence'"
            ).fetchone():
                conn.execute("DELETE FROM sqlite_sequence;")
        conn.execute("VACUUM;")

        print(f"Cleared {total} row(s) across {len(tables)} table(s).")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
