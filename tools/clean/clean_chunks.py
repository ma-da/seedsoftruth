#!/usr/bin/env python3
"""
Clean low-value sentences from chunks.fulltext_text in the hybrid RAG DB.

Iterates every row in `chunks`, splits its fulltext into sentences, drops
any sentence that matches one of the configured low-value patterns
(navigation chrome, calls-to-action, cookie banners, donate/subscribe
prompts, copyright lines, WantToKnow-specific site chrome, bare-URL
lines, etc.), and re-stitches what's left back into the row.

DEFAULT IS DRY-RUN. Nothing is written to the DB unless you pass
--apply. With --apply, a timestamped .bak.<TS> copy of the DB file is
created up-front (unless --no-backup is set, which it ignores by
prompting for confirmation).

Run:
  python3 tools/clean/clean_chunks.py                          # dry-run, full DB
  python3 tools/clean/clean_chunks.py --limit 500              # quick preview
  python3 tools/clean/clean_chunks.py --apply                  # writes (with backup)
  python3 tools/clean/clean_chunks.py --apply --no-backup      # skip backup (with confirm)
  python3 tools/clean/clean_chunks.py --apply --no-fts-rebuild # skip the FTS5 rebuild

Stats printed at end:
  - Chunks examined / modified
  - Sentences removed (total + per pattern, with examples)
  - Total characters kept / removed (% size reduction)
  - Elapsed time

The script never touches the entity_fts virtual table (entity index is
keyed off entities, not fulltext_text — cleaning the fulltext leaves it
correct). It DOES rebuild fulltext_fts at the end so the BM25 index
matches the cleaned text. If your fulltext_fts is trigger-synced rather
than rebuild-able, the rebuild call will be skipped with a logged note.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


# ------------------------ Patterns ------------------------
#
# Each pattern is (name, compiled_regex). A sentence is dropped if the
# regex matches anywhere in it. Names are used in the stats report so
# you can see which patterns are firing most often and tune from there.
#
# Conservative bias: only drop sentences that are clearly chrome /
# CTAs / boilerplate. We err toward keeping content — false-positive
# drops would hurt retrieval quality more than a little surviving
# chrome does.

_pat = lambda name, regex: (name, re.compile(regex, re.IGNORECASE))

PATTERNS: List[Tuple[str, re.Pattern]] = [
    # ---- Calls to action that imply linkified UI ----
    _pat("click_here",
         r"\bclick\s+(?:here|below|to\s+(?:read|learn|view|download|continue|sign(?:\s+up)?|subscribe))\b"),
    _pat("tap_or_press_link",
         r"\b(?:tap|press)\s+(?:here|the\s+link)\b"),
    _pat("read_more_cta",
         r"\bread\s+(?:more|the\s+full\s+(?:article|story|post))\b"),

    # ---- Social / share ----
    _pat("share_on",
         r"\bshare\s+(?:on|via|this\s+(?:article|post|story|page))\b"),
    _pat("tweet_this",
         r"\btweet\s+(?:this|to\s+(?:share|tell))\b"),
    _pat("follow_us",
         r"\b(?:follow|like)\s+us\s+on\s+(?:facebook|twitter|x|instagram|linkedin|youtube|telegram|tiktok|substack)\b"),

    # ---- Subscribe / signup / donate / support ----
    _pat("subscribe_cta",
         r"\bsubscribe\s+(?:to\s+(?:our\s+)?(?:newsletter|youtube|channel|substack|mailing\s+list|email)|now|today|for\s+(?:free|updates))\b"),
    _pat("signup_cta",
         r"\bsign\s+up\s+(?:for|to\s+(?:our|receive)|now|today)\b"),
    _pat("get_our_newsletter",
         r"\b(?:get|receive)\s+(?:our\s+)?(?:free\s+)?(?:weekly|daily|monthly)?\s*(?:newsletter|updates|emails)\b"),
    _pat("donate_cta",
         r"\bdonate\s+(?:now|today|here|to\s+support|via\s+(?:paypal|venmo))\b"),
    _pat("support_us",
         r"\bsupport\s+(?:our|this|independent)\s+(?:work|journalism|site|publication|reporting|mission)\b"),

    # ---- Legal / privacy / cookie chrome ----
    _pat("copyright",
         r"©\s*\d{4}|\(c\)\s*\d{4}|\ball\s+rights\s+reserved\b"),
    _pat("cookie_banner",
         r"\bthis\s+(?:site|website|page)\s+uses\s+cookies\b|\baccept\s+(?:all\s+)?cookies\b|\bcookie\s+(?:policy|settings|preferences)\b"),
    _pat("privacy_terms_link",
         r"\b(?:privacy|cookie)\s+(?:policy|notice)\b|\bterms\s+(?:of\s+(?:service|use)|and\s+conditions)\b"),

    # ---- Comment / forum chrome ----
    _pat("leave_comment",
         r"\bleave\s+(?:a\s+)?(?:comment|reply)\b"),
    _pat("filed_under",
         r"\b(?:filed\s+under|tagged\s+(?:with|as))\b"),

    # ---- Navigation crumbs ----
    _pat("back_to",
         r"\bback\s+to\s+(?:top|home|index|main|the\s+top)\b"),
    _pat("next_prev_page",
         r"\b(?:next|previous|prev)\s+(?:post|article|page|chapter|story)\b"),

    # ---- Bare URL on its own ----
    _pat("bare_url_line",
         r"^\s*(?:https?://\S+|www\.[^\s]+)\s*$"),

    # ---- Disclaimers commonly tacked on at end of articles ----
    _pat("not_advice",
         r"\bnot\s+(?:medical|legal|financial|investment|professional)\s+advice\b"),
    _pat("informational_only",
         r"\bfor\s+(?:informational|educational|entertainment)\s+purposes\s+only\b"),
    _pat("disclaimer_prefix",
         r"^\s*disclaimer\s*:\s*"),

    # ---- WantToKnow.info-specific site chrome (observed at end of WTK chunks) ----
    _pat("wtk_site_serves",
         r"\bour\s+site\s+serves\s+as\s+a\s+research\s+tool\s+and\s+comprehensive\s+archive\b"),
    _pat("wtk_inspiration_center",
         r"\bour\s+Inspiration\s+Center\s+seeks\b"),
    _pat("wtk_summarized_count",
         r"\bwe[''']?ve\s+summarized\s+over\s+\d[\d,]*\s+(?:inspiring\s+)?news\s+articles\b"),
    _pat("wtk_what_you_can_do",
         r"\bcheck\s+out\s+our\s+[\"\']?what\s+you\s+can\s+do[\"\']?\s+section\b"),
    _pat("wtk_book_in_entirety",
         r"\bavailable\s+in\s+its\s+entirety\s+on\s+this\s+webpage\b"),

    # ---- Image / media credit boilerplate ----
    _pat("image_credit",
         r"^\s*(?:image|photo|illustration|video)\s+(?:credit|source|by)\s*[:\-]"),
    _pat("caption_prefix",
         r"^\s*(?:caption|figure|fig\.)\s*[:\-]"),
]


# ------------------------ Sentence splitting ------------------------
#
# Pragmatic splitter: end-of-sentence punctuation followed by whitespace,
# OR any newline. Not perfect on abbreviations ("Dr.", "U.S."), but
# for filtering chrome we only need approximate boundaries — false
# splits inside content sentences still produce keep-able fragments.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_WS_RUN = re.compile(r"\s+")


def split_sentences(text: str) -> List[str]:
    """Split text into approximate sentences on end punctuation or newlines.

    Args:
        text: The text to split.

    Returns:
        A list of whitespace-stripped, non-empty sentence fragments. Empty
        if the input is empty.
    """
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


# ------------------------ Cleaning ------------------------

def clean_text(text: str) -> Tuple[str, Counter, List[Tuple[str, str]]]:
    """
    Returns (cleaned_text, pattern_hit_counter, removed_sentences).
    `removed_sentences` is a list of (pattern_name, sentence) for the
    in-process stats report.
    """
    if not text:
        return text, Counter(), []

    sentences = split_sentences(text)
    kept: List[str] = []
    removed: List[Tuple[str, str]] = []
    hits: Counter = Counter()

    for s in sentences:
        matched = None
        for name, rgx in PATTERNS:
            if rgx.search(s):
                matched = name
                break
        if matched is not None:
            hits[matched] += 1
            removed.append((matched, s))
        else:
            kept.append(s)

    # Re-stitch. We've lost original spacing (the split discarded the
    # separator), so join with single spaces and collapse runs.
    cleaned = " ".join(kept)
    cleaned = _WS_RUN.sub(" ", cleaned).strip()
    return cleaned, hits, removed


# ------------------------ DB plumbing ------------------------

def backup_db(db_path: Path) -> Path:
    """Copy the DB file to a timestamped ``.bak.<TS>`` sibling.

    Args:
        db_path: Path to the SQLite DB file to back up.

    Returns:
        The path of the created backup file.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = db_path.with_suffix(db_path.suffix + f".bak.{ts}")
    print(f"Backing up {db_path}")
    print(f"     → {bak}")
    shutil.copy2(db_path, bak)
    size_mb = bak.stat().st_size / (1024 * 1024)
    print(f"  Backup created ({size_mb:,.1f} MB)")
    return bak


def iterate_chunks(conn: sqlite3.Connection, limit: int | None):
    """
    Yield (lookup_id, fulltext_text) pairs. Streams via a server-side
    cursor so we don't pull the whole table into memory.
    """
    cur = conn.cursor()
    if limit:
        cur.execute("SELECT lookup_id, fulltext_text FROM chunks LIMIT ?", (limit,))
    else:
        cur.execute("SELECT lookup_id, fulltext_text FROM chunks")
    for row in cur:
        yield row[0], row[1]


def _split_top_level_commas(s: str) -> List[str]:
    """Split `s` on commas that are not inside single/double quotes."""
    parts: List[str] = []
    cur: List[str] = []
    in_quote: str | None = None
    for ch in s:
        if in_quote:
            cur.append(ch)
            if ch == in_quote:
                in_quote = None
        elif ch in ("'", '"'):
            in_quote = ch
            cur.append(ch)
        elif ch == ",":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def fts_schema_info(conn: sqlite3.Connection, table_name: str) -> dict | None:
    """
    Parse the CREATE VIRTUAL TABLE statement for an FTS5 table and
    return its type / columns / content source. Returns None if the
    table isn't an FTS5 virtual table (or doesn't exist).

    Returned dict keys:
      type           : 'standalone' | 'external_content' | 'contentless'
      columns        : list of indexed column names (excludes directives)
      content_table  : str | None — name of the source table for
                       external-content FTS5; None for standalone /
                       contentless.

    "standalone" means the FTS5 table stores its own copy of the
    content (no `content=` clause). Updating the source table does
    NOT update the FTS — `'rebuild'` only compacts segments.
    """
    cur = conn.cursor()
    row = cur.execute(
        "SELECT sql FROM sqlite_master WHERE name = ? AND type = 'table'",
        (table_name,),
    ).fetchone()
    if not row or not row[0]:
        return None
    sql = row[0]
    sql_lc = sql.lower()
    if "fts5" not in sql_lc or "virtual table" not in sql_lc:
        return None

    # content= directive
    content_table: str | None = None
    fts_type = "standalone"
    m = re.search(r"content\s*=\s*'([^']*)'", sql, flags=re.I)
    if m:
        if m.group(1) == "":
            fts_type = "contentless"
        else:
            fts_type = "external_content"
            content_table = m.group(1)
    else:
        m2 = re.search(r"content\s*=\s*([A-Za-z_]\w*)", sql, flags=re.I)
        if m2:
            fts_type = "external_content"
            content_table = m2.group(1)

    # Extract the column list from inside `fts5( ... )`.
    body_m = re.search(r"fts5\s*\(\s*(.*)\s*\)", sql, flags=re.I | re.S)
    columns: List[str] = []
    if body_m:
        body = body_m.group(1)
        for part in _split_top_level_commas(body):
            part = part.strip().rstrip(",").strip()
            if not part:
                continue
            # Skip directive-only parts (tokenize=, content=, content_rowid=,
            # prefix=, columnsize=, detail=, etc.)
            if re.match(r"^[A-Za-z_]\w*\s*=", part):
                continue
            # Column name optionally followed by UNINDEXED, possibly quoted.
            col_m = re.match(
                r"^[\"']?([A-Za-z_]\w*)[\"']?(?:\s+UNINDEXED)?\s*$",
                part, flags=re.I,
            )
            if col_m:
                columns.append(col_m.group(1))

    return {
        "type": fts_type,
        "columns": columns,
        "content_table": content_table,
    }


def fts_sync(
    conn: sqlite3.Connection,
    fts_table: str,
    source_table: str,
    source_rowid_col: str,
) -> bool:
    """
    Bring the FTS5 index up to date with `source_table`. Picks the
    right primitive based on the FTS table's mode:

      standalone        — DELETE + re-INSERT every row (the only way;
                           'rebuild' just compacts segments).
      external_content  — INSERT INTO fts(fts) VALUES('rebuild') re-reads
                           from the source table.
      contentless       — neither approach works without app-level
                           triggers; warn and skip.

    Returns True if a sync was performed, False if skipped.
    """
    info = fts_schema_info(conn, fts_table)
    if info is None:
        print(f"  {fts_table} is not an FTS5 virtual table — skipping sync.")
        return False

    print(f"  {fts_table} mode: {info['type']}")
    print(f"  columns: {info['columns']}")

    if info["type"] == "external_content":
        try:
            conn.execute(
                f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')"
            )
            conn.commit()
            print("  external-content rebuild OK.")
            return True
        except sqlite3.OperationalError as e:
            print(f"  external-content rebuild failed: {e}")
            return False

    if info["type"] == "contentless":
        print("  contentless FTS — can't resync from a source table. "
              "Triggers in the app are required to keep it in sync.")
        return False

    # ---- Standalone: wipe + reinsert from source_table ----
    cols = info["columns"]
    if not cols:
        print(f"  {fts_table} has no parseable columns — skipping sync.")
        return False

    # Verify source_table has every FTS column. If not, abort rather
    # than write garbage.
    src_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({source_table})")}
    missing = [c for c in cols if c not in src_cols]
    if missing:
        print(f"  source table {source_table} is missing FTS column(s): "
              f"{missing}. Skipping sync.")
        return False
    if source_rowid_col not in src_cols:
        print(f"  source table {source_table} has no column "
              f"'{source_rowid_col}' to use as FTS rowid. Skipping sync.")
        return False

    col_list = ", ".join(cols)
    print(f"  wiping {fts_table} ...")
    conn.execute(f"DELETE FROM {fts_table}")
    print(f"  re-inserting from {source_table} (this may take a few minutes) ...")
    t0 = time.time()
    conn.execute(
        f"INSERT INTO {fts_table}(rowid, {col_list}) "
        f"SELECT {source_rowid_col}, {col_list} FROM {source_table}"
    )
    conn.commit()
    print(f"  standalone resync OK ({time.time() - t0:.1f}s).")
    return True


def run_vacuum(db_path: Path) -> None:
    """
    VACUUM rewrites the file packed and reclaims free pages. Needs a
    fresh autocommit-mode connection (no open transactions) and roughly
    DB-size of free disk space while it runs.
    """
    print(f"VACUUM {db_path} ...")
    t0 = time.time()
    # isolation_level=None → autocommit. VACUUM can't run inside a tx.
    vac = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        vac.execute("VACUUM")
    finally:
        vac.close()
    print(f"  VACUUM done ({time.time() - t0:.1f}s).")


# ------------------------ Main ------------------------

def main(argv: List[str] | None = None) -> int:
    """Parse args and run the chunk-cleaning (and optional FTS/VACUUM) pipeline.

    Resolves the DB path, optionally backs it up, then either resyncs the FTS
    index only (``--fts-only``) or streams every chunk through ``clean_text``,
    applying UPDATEs in batches when ``--apply`` is set, optionally rebuilding
    the FTS index and running VACUUM. Prints a per-pattern summary report.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when None.

    Returns:
        Process exit code: 0 on success, 1 if the user aborts a no-backup
        apply or misuses ``--fts-only``, 2 if the DB file is missing.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db", type=Path,
        default=Path("data/gamma_master_hybrid_fts_stage3.db"),
        help="Path to the hybrid RAG SQLite DB",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually write changes. Without this, runs in dry-run mode.",
    )
    ap.add_argument(
        "--no-backup", action="store_true",
        help="Skip the timestamped backup when --apply. Will prompt to confirm.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of rows to process (useful for dry-run preview).",
    )
    ap.add_argument(
        "--batch", type=int, default=500,
        help="UPDATE batch size when --apply (rows per commit).",
    )
    ap.add_argument(
        "--examples", type=int, default=3,
        help="How many sentence examples to show per pattern in the report.",
    )
    ap.add_argument(
        "--no-fts-rebuild", action="store_true",
        help="Skip the FTS5 sync after applying. Default is to sync.",
    )
    ap.add_argument(
        "--fts-table", default="fulltext_fts",
        help="Name of the FTS5 virtual table to keep in sync.",
    )
    ap.add_argument(
        "--source-table", default="chunks",
        help="Source table the FTS5 index mirrors.",
    )
    ap.add_argument(
        "--rowid-col", default="lookup_id",
        help="Column in --source-table to use as the FTS rowid.",
    )
    ap.add_argument(
        "--vacuum", action="store_true",
        help="Run VACUUM after applying changes. Reclaims free pages "
             "and shrinks the file. Needs ~DB-size of free disk space.",
    )
    ap.add_argument(
        "--fts-only", action="store_true",
        help="Skip the chunk cleanup entirely and only resync the FTS "
             "table. Useful for fixing a previous run that left FTS stale.",
    )
    args = ap.parse_args(argv)

    db_path = args.db
    if not db_path.is_absolute():
        db_path = (Path(__file__).resolve().parent.parent / db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: db not found at {db_path}", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode     : {mode}")
    print(f"DB       : {db_path}")
    print(f"Limit    : {args.limit if args.limit else 'all rows'}")
    print(f"Batch    : {args.batch}")
    print(f"Patterns : {len(PATTERNS)}")
    print()

    # Backup logic.
    if args.apply and not args.no_backup:
        backup_db(db_path)
        print()
    elif args.apply and args.no_backup:
        print("WARNING: --apply --no-backup will write to the DB without a snapshot.")
        try:
            ans = input("Type 'YES' to proceed: ").strip()
        except EOFError:
            ans = ""
        if ans != "YES":
            print("Aborted.")
            return 1
        print()

    conn = sqlite3.connect(str(db_path), isolation_level="DEFERRED")
    cur = conn.cursor()

    # --fts-only: skip the chunk-cleaning loop, just sync FTS (and
    # optionally VACUUM). Useful when a previous run cleaned chunks
    # but didn't update the FTS — apply mode is required.
    if args.fts_only:
        if not args.apply:
            print("--fts-only requires --apply (it's a write operation).")
            conn.close()
            return 1
        print(f"FTS-only mode: syncing {args.fts_table} ←  {args.source_table}")
        fts_sync(conn, args.fts_table, args.source_table, args.rowid_col)
        conn.close()
        if args.vacuum:
            print()
            run_vacuum(db_path)
        print()
        print("Done.")
        return 0

    total_chunks = 0
    modified_chunks = 0
    empty_chunks = 0
    total_removed_sentences = 0
    total_kept_chars = 0
    total_removed_chars = 0
    pattern_hits: Counter = Counter()
    pattern_examples: dict[str, List[str]] = {name: [] for name, _ in PATTERNS}
    pending_updates: List[Tuple[str, int]] = []

    t0 = time.time()
    try:
        for lookup_id, text in iterate_chunks(conn, args.limit):
            total_chunks += 1
            if text is None or not str(text).strip():
                empty_chunks += 1
                continue

            original_len = len(text)
            cleaned, hits, removed = clean_text(text)

            if hits and cleaned != text:
                modified_chunks += 1
                pattern_hits.update(hits)
                total_removed_sentences += len(removed)
                total_kept_chars += len(cleaned)
                total_removed_chars += (original_len - len(cleaned))

                for name, sent in removed:
                    if len(pattern_examples[name]) < args.examples:
                        pattern_examples[name].append(sent[:200])

                if args.apply:
                    pending_updates.append((cleaned, lookup_id))
                    if len(pending_updates) >= args.batch:
                        cur.executemany(
                            "UPDATE chunks SET fulltext_text = ? WHERE lookup_id = ?",
                            pending_updates,
                        )
                        conn.commit()
                        pending_updates.clear()
            else:
                total_kept_chars += original_len

            if total_chunks % 5000 == 0:
                elapsed = time.time() - t0
                rate = total_chunks / max(elapsed, 0.001)
                print(f"  ... {total_chunks:,} examined "
                      f"({modified_chunks:,} modified, {rate:.0f}/s)")

        # Flush any pending updates.
        if args.apply and pending_updates:
            cur.executemany(
                "UPDATE chunks SET fulltext_text = ? WHERE lookup_id = ?",
                pending_updates,
            )
            conn.commit()
            pending_updates.clear()

        # FTS5 sync — only meaningful if we actually changed data.
        # Uses the type-aware fts_sync that handles standalone /
        # external_content / contentless FTS5 correctly.
        if args.apply and not args.no_fts_rebuild and modified_chunks > 0:
            print()
            print(f"Syncing FTS index ({args.fts_table}) ...")
            fts_sync(conn, args.fts_table, args.source_table, args.rowid_col)

    finally:
        conn.close()

    # VACUUM, if requested. Must run AFTER the connection is closed so
    # SQLite can hold the exclusive lock it needs.
    if args.apply and args.vacuum and modified_chunks > 0:
        print()
        run_vacuum(db_path)

    elapsed = time.time() - t0

    # ---------- Report ----------
    print()
    print("=" * 72)
    print(" Summary")
    print("=" * 72)
    print(f"Mode                  : {mode}")
    print(f"Chunks examined       : {total_chunks:,}")
    if total_chunks:
        pct = 100.0 * modified_chunks / total_chunks
        print(f"Chunks modified       : {modified_chunks:,} ({pct:.1f}% of examined)")
    print(f"Empty chunks skipped  : {empty_chunks:,}")
    print(f"Sentences removed     : {total_removed_sentences:,}")
    print(f"Chars kept            : {total_kept_chars:,}")
    print(f"Chars removed         : {total_removed_chars:,}")
    total_chars = total_kept_chars + total_removed_chars
    if total_chars:
        red_pct = 100.0 * total_removed_chars / total_chars
        print(f"Net text reduction    : {red_pct:.2f}%")
    print(f"Elapsed               : {elapsed:.1f}s")

    print()
    print("Per-pattern hits (sorted desc):")
    if not pattern_hits:
        print("  (no patterns matched any sentence)")
    else:
        for name, n in pattern_hits.most_common():
            print(f"  {n:8,d}  {name}")
            for ex in pattern_examples[name]:
                print(f"            ex: {ex}")

    if not args.apply:
        print()
        print("DRY-RUN: no changes written. Re-run with --apply to commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
