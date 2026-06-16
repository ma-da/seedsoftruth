#!/usr/bin/env python3
"""
Sample chunks from the hybrid retrieval DB and surface candidate
sentence/line patterns to filter out during chunk cleaning.

Run:
  python3 tools/eval/analyze_chunks.py
  python3 tools/eval/analyze_chunks.py --sample 250 --db data/gamma_master_hybrid_fts_stage3.db
  python3 tools/eval/analyze_chunks.py --sample 100 --raw-out /tmp/chunks_raw.txt

What it does:
1. Selects N random rows from the `chunks` table (deterministic when --seed is set).
2. For each row's `fulltext_text` it tokenizes into "candidate lines"
   (period-delimited sentences plus newline-delimited lines, deduped).
3. Runs every line through a battery of heuristic pattern detectors —
   keyword phrases ("click here", "share on facebook"), structural
   signals (mostly-punctuation, URL-only, very short, repeated symbols),
   and known web-chrome cues (copyright lines, cookie banners, etc.).
4. Prints, sorted by frequency:
     - exact recurring lines (the dumbest, most-useful filter target)
     - matched-keyword counts (per pattern)
     - top "all-uppercase shouty" lines (often headers / nav)
     - high-punctuation-ratio lines (often menus / breadcrumbs)
5. Optionally dumps the raw fulltext samples to --raw-out so you can
   eyeball ground truth and refine the heuristics.

Output is read-only — the script never writes to the DB.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

# --------------------- Heuristic vocabulary ---------------------
#
# These are seed phrases known to recur in scraped web corpora as
# navigation / share / promo / cookie chrome rather than substantive
# content. The list is intentionally conservative — recall over
# precision. The frequency report at the end tells you which ones are
# actually worth filtering for *this* corpus.
KEYWORD_PATTERNS = [
    # Calls-to-action that imply linkified UI
    r"\bclick (?:here|below|to read|to learn|to view|to download)\b",
    r"\bread (?:more|the full article|on)\b",
    r"\b(?:learn|find out) more\b",
    r"\b(?:download|view|see) (?:the )?(?:pdf|video|gallery)\b",
    r"\b(?:tap|press) (?:here|the link)\b",

    # Social share / engagement
    r"\bshare (?:on|via|this)\b",
    r"\btweet (?:this|@)\b",
    r"\b(?:follow|like) us on\b",
    r"\b(?:facebook|twitter|x\.com|linkedin|reddit|pinterest|telegram|whatsapp|email|print)\b\s*(?:share|button|link)?",
    r"\bsubscribe (?:to (?:our )?(?:newsletter|youtube|channel|substack)|now)\b",

    # Promo / paywall
    r"\b(?:subscribe|sign up|join|become a member)\b.*\b(?:today|now|free|here)\b",
    r"\b(?:get|receive) (?:our )?(?:free|weekly|daily|monthly) (?:newsletter|updates|emails)\b",
    r"\bsupport (?:our|this) (?:work|journalism|site|publication)\b",
    r"\bdonate (?:now|today|here)\b",
    r"\b(?:patreon|paypal|venmo|bitcoin)\b",

    # Legal / privacy / cookie chrome
    r"\ball rights reserved\b",
    r"\b(?:©|\(c\))\s*\d{4}",
    r"\b(?:privacy|cookie) (?:policy|notice|settings)\b",
    r"\bterms (?:of (?:service|use)|and conditions)\b",
    r"\bthis (?:site|website) uses cookies\b",
    r"\b(?:accept|reject) (?:cookies|all)\b",

    # Comment / forum chrome
    r"\bleave (?:a )?(?:comment|reply)\b",
    r"\b\d+\s+(?:comment|reply|views|likes|shares|reads)\b",
    r"\b(?:posted|published|written) (?:by|on)\b\s+(?:[A-Z][a-z]+\s*){0,3}",
    r"\bfiled under\b",
    r"\btagged (?:with|as)\b",

    # Navigation crumbs / TOC
    r"\b(?:home|news|about|contact|search|archive|categories?|menu|sitemap)\b\s*(?:»|>|\|)",
    r"\b(?:next|previous|prev) (?:post|article|page|chapter)\b",
    r"\bback to (?:top|home|index|main)\b",
    r"\btable of contents\b",

    # Image / media credits
    r"\b(?:image|photo|illustration|video) (?:credit|source|by)\b",
    r"\b(?:caption|figure|fig\.)\s*[:\-]",
    r"\b(?:source|via)\s*:\s*",

    # Boilerplate disclaimers
    r"\bdisclaimer\s*:\s*",
    r"\bthe views (?:expressed|are those)\b",
    r"\bnot (?:medical|legal|financial|investment) advice\b",
    r"\bfor (?:informational|educational) purposes only\b",
]

KEYWORD_REGEXES = [re.compile(p, re.IGNORECASE) for p in KEYWORD_PATTERNS]


# --------------------- Line splitting ---------------------

# Split on newlines and on sentence-ending punctuation. Not perfect
# (abbreviations etc.) but good enough for boilerplate-hunting.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def split_lines(text: str) -> List[str]:
    """Split text into candidate lines on sentence punctuation and newlines.

    Args:
        text: The text to split.

    Returns:
        A list of whitespace-stripped, non-empty fragments; empty if the
        input is empty.
    """
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


# --------------------- Structural scoring ---------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9\s]")


def punct_ratio(line: str) -> float:
    """Fraction of chars that are punctuation/symbols (excl. whitespace)."""
    stripped = line.replace(" ", "")
    if not stripped:
        return 0.0
    non_alnum = _NON_ALNUM_RE.findall(stripped)
    return len(non_alnum) / len(stripped)


def url_share(line: str) -> float:
    """Fraction of the line's length consumed by URLs."""
    if not line:
        return 0.0
    urls = _URL_RE.findall(line)
    if not urls:
        return 0.0
    return sum(len(u) for u in urls) / len(line)


def is_shouty(line: str) -> bool:
    """True if the line is mostly uppercase letters (>=6 letters, >=80% upper)."""
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 6:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) >= 0.80


def normalize_for_dedupe(line: str) -> str:
    """Collapse whitespace and lowercase so near-duplicates merge."""
    return re.sub(r"\s+", " ", line.strip().lower())


# --------------------- Sampling ---------------------

def sample_rows(db_path: Path, sample_n: int, seed: int | None) -> List[sqlite3.Row]:
    """Fetch a random sample of chunk rows from the DB.

    Args:
        db_path: Path to the hybrid retrieval SQLite DB.
        sample_n: Number of rows to return.
        seed: If set, use a stable hash-based ordering for reproducibility;
            otherwise order by SQLite ``RANDOM()``.

    Returns:
        A list of ``sqlite3.Row`` objects, one per sampled chunk.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Seed RANDOM() deterministically when requested. SQLite doesn't
    # expose a seedable PRNG, so when seed is set we use a stable
    # hash-based ordering instead of RANDOM().
    if seed is not None:
        cur.execute(
            """
            SELECT lookup_id, chunk_id, title, subset_name, domain,
                   fulltext_text, source_url
            FROM chunks
            ORDER BY (lookup_id * ?) % 1000003
            LIMIT ?
            """,
            (seed, sample_n),
        )
    else:
        cur.execute(
            """
            SELECT lookup_id, chunk_id, title, subset_name, domain,
                   fulltext_text, source_url
            FROM chunks
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (sample_n,),
        )
    rows = cur.fetchall()
    conn.close()
    return rows


# --------------------- Analysis ---------------------

def analyze(rows: Iterable[sqlite3.Row]) -> dict:
    """Run the heuristic boilerplate-detection battery over sampled chunks.

    Tokenizes each chunk's fulltext into candidate lines and tallies exact
    recurring lines, keyword-pattern hits (with examples), shouty/uppercase
    lines, short recurring lines, high-punctuation lines, and URL-dominated
    lines.

    Args:
        rows: Iterable of chunk rows (each must have ``fulltext_text``).

    Returns:
        A dict of aggregated counters/lists keyed by signal name, plus
        ``total_chunks`` and ``total_lines`` totals, consumed by ``report``.
    """
    line_freq: Counter[str] = Counter()  # normalized → count of exact recurrence
    keyword_hits: Counter[str] = Counter()  # pattern → match count
    keyword_examples: dict[str, List[str]] = {p: [] for p in KEYWORD_PATTERNS}
    shouty_lines: Counter[str] = Counter()
    high_punct_lines: List[Tuple[float, str]] = []
    short_lines: Counter[str] = Counter()
    url_dominated: Counter[str] = Counter()
    total_chunks = 0
    total_lines = 0

    for row in rows:
        text = row["fulltext_text"] or ""
        if not text.strip():
            continue
        total_chunks += 1
        lines = split_lines(text)
        for line in lines:
            total_lines += 1
            norm = normalize_for_dedupe(line)

            # Frequency of exact (normalized) line — strongest filter signal
            line_freq[norm] += 1

            # Structural buckets
            if len(line) <= 25 and any(c.isalpha() for c in line):
                short_lines[norm] += 1
            if is_shouty(line):
                shouty_lines[norm] += 1
            pr = punct_ratio(line)
            if pr >= 0.35 and len(line) >= 8:
                high_punct_lines.append((pr, line))
            if url_share(line) >= 0.5:
                url_dominated[norm] += 1

            # Keyword hits
            for pat, rgx in zip(KEYWORD_PATTERNS, KEYWORD_REGEXES):
                if rgx.search(line):
                    keyword_hits[pat] += 1
                    if len(keyword_examples[pat]) < 3:
                        keyword_examples[pat].append(line[:160])

    return {
        "total_chunks": total_chunks,
        "total_lines": total_lines,
        "line_freq": line_freq,
        "keyword_hits": keyword_hits,
        "keyword_examples": keyword_examples,
        "shouty_lines": shouty_lines,
        "high_punct_lines": high_punct_lines,
        "short_lines": short_lines,
        "url_dominated": url_dominated,
    }


# --------------------- Reporting ---------------------

def print_section(title: str) -> None:
    """Print a boxed section header to stdout."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def report(result: dict, top: int = 30) -> None:
    """Print the full human-readable analysis report to stdout.

    Args:
        result: The dict returned by ``analyze``.
        top: How many items to show per ranked section.
    """
    print_section("Summary")
    print(f"Sampled chunks      : {result['total_chunks']}")
    print(f"Total candidate lines: {result['total_lines']}")

    print_section(f"Top {top} exact recurring lines (best literal-filter candidates)")
    print("Lines that appear more than once across the sample are almost")
    print("always boilerplate. The count is how many chunks they showed up in.")
    print()
    repeats = [(line, n) for line, n in result["line_freq"].most_common() if n >= 2]
    for line, n in repeats[:top]:
        print(f"  {n:4d}  {line[:140]}")
    if not repeats:
        print("  (no lines recurred — sample may be too small or corpus very diverse)")

    print_section("Keyword-pattern hit counts")
    print("How many lines matched each seed regex. High counts = worth")
    print("filtering on; zero counts may indicate the pattern doesn't")
    print("apply to this corpus.")
    print()
    for pat, n in sorted(result["keyword_hits"].items(), key=lambda x: -x[1]):
        if n == 0:
            continue
        print(f"  {n:5d}  /{pat}/i")
        for ex in result["keyword_examples"].get(pat, [])[:2]:
            print(f"          ex: {ex}")

    print_section(f"Top {top} 'shouty' (mostly-uppercase) lines")
    print("Often headers, nav labels, or ALL-CAPS calls-to-action.")
    print()
    for line, n in result["shouty_lines"].most_common(top):
        print(f"  {n:4d}  {line[:140]}")
    if not result["shouty_lines"]:
        print("  (none)")

    print_section(f"Top {top} short recurring lines (≤25 chars)")
    print("These are usually buttons, menu items, or breadcrumbs.")
    print()
    for line, n in result["short_lines"].most_common(top):
        if n < 2:
            continue
        print(f"  {n:4d}  {line[:80]}")

    print_section("High-punctuation-ratio sample (top 20)")
    print("Lines with ≥35% non-alphanumeric characters — likely menus,")
    print("dashes, breadcrumbs, or noise.")
    print()
    for ratio, line in sorted(result["high_punct_lines"], reverse=True)[:20]:
        print(f"  {ratio*100:5.1f}%  {line[:140]}")

    print_section(f"Top {top} URL-dominated lines")
    print("Lines that are ≥50% URL by character count. Usually safe to")
    print("strip entirely from the chunk text.")
    print()
    for line, n in result["url_dominated"].most_common(top):
        print(f"  {n:4d}  {line[:120]}")


def write_raw_samples(rows: Iterable[sqlite3.Row], out_path: Path) -> int:
    """Dump the raw fulltext of each sampled chunk to a file for eyeballing.

    Args:
        rows: Iterable of chunk rows to write.
        out_path: File to write the formatted dump to.

    Returns:
        The number of rows written.
    """
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"=== lookup_id={row['lookup_id']} chunk_id={row['chunk_id']} "
                    f"subset={row['subset_name']} domain={row['domain']} ===\n")
            f.write(f"TITLE: {row['title']}\n")
            f.write(f"URL: {row['source_url']}\n")
            f.write("FULLTEXT:\n")
            f.write((row["fulltext_text"] or "") + "\n")
            f.write("\n\n")
            n += 1
    return n


# --------------------- CLI ---------------------

def main(argv: List[str] | None = None) -> int:
    """Parse args, sample chunks, and print the boilerplate-analysis report.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when None.

    Returns:
        Process exit code: 0 on success, 2 if the DB is missing, 3 if the
        chunks table returned no rows.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("data/gamma_master_hybrid_fts_stage3.db"),
        help="Path to the hybrid retrieval SQLite DB",
    )
    ap.add_argument("--sample", type=int, default=100, help="How many rows to sample")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, use a stable hash-based ordering for reproducibility",
    )
    ap.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Optional path to dump the raw sampled fulltext for eyeballing",
    )
    ap.add_argument("--top", type=int, default=30, help="How many items per report section")
    args = ap.parse_args(argv)

    db_path = args.db
    if not db_path.is_absolute():
        # Resolve relative to the repo root (parent of tools/)
        db_path = (Path(__file__).resolve().parent.parent / db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: db not found at {db_path}", file=sys.stderr)
        return 2

    rows = sample_rows(db_path, args.sample, args.seed)
    if not rows:
        print("ERROR: no rows returned from chunks table", file=sys.stderr)
        return 3

    print(f"Loaded {len(rows)} rows from {db_path}")

    if args.raw_out:
        # Re-pull or reuse rows; we already have them.
        n = write_raw_samples(rows, args.raw_out)
        print(f"Wrote {n} raw samples to {args.raw_out}")

    result = analyze(rows)
    report(result, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
