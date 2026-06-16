#!/usr/bin/env python3
"""
filter_corpus.py
================

File-level quality gate for a raw text corpus. For every input file:

1.  Read with encoding sniffing (`chardet`), falling back to utf-8.
2.  If the file looks like HTML (or has a `.html`/`.htm` extension),
    strip tags with BeautifulSoup.
3.  Fix mojibake with `ftfy`, NFC-normalize, drop non-printables, and
    replace punctuation runs with spaces (aggressive — produces a
    single-line "tokenizable" version used ONLY for the quality check).
4.  Decide keep/reject:
        * `has_natural_language_run` — at least one 60-word sliding
          window where ≥80% of tokens are alphabetic AND the average
          per-token non-ASCII fraction is ≤20%.
        * `is_garbled` — overall letter-to-nonspace ratio must be
          ≥55% and the cleaned text must be at least 400 chars long.
    KEEP if both pass; otherwise REJECT.
5.  Kept files are written to `--output-dir` as `.txt` (preserving the
    relative path under `--input-dir`).
6.  Rejected files' originals are MOVED to `--reject-dir`. Pass
    `--copy-rejects` to copy instead of move, or `--dry-run` to do
    neither and just log decisions.
7.  A CSV decision log is written to `--log` (default
    `<output-dir>/clean_log.csv`) with columns:
    `source_path, decision, clean_chars, total_words`.

Use this BEFORE `clean_web_corpus.py` (which assumes its inputs are
already roughly natural-language text). The output of this script is
a directory of files that have passed the quality gate but have NOT
yet had boilerplate / TOC / index removed.

Usage
-----
    # Dry run, no writes:
    python tools/clean/filter_corpus.py \\
        --input-dir  /data/raw_corpus \\
        --output-dir /data/filtered_corpus \\
        --reject-dir /data/rejected \\
        --dry-run

    # Actual run:
    python tools/clean/filter_corpus.py \\
        --input-dir  /data/raw_corpus \\
        --output-dir /data/filtered_corpus \\
        --reject-dir /data/rejected

    # Custom thresholds:
    python tools/clean/filter_corpus.py \\
        --input-dir  /data/raw_corpus \\
        --output-dir /data/filtered_corpus \\
        --reject-dir /data/rejected \\
        --window-size 100 \\
        --alpha-min 0.85 \\
        --max-nonascii 0.15 \\
        --garbled-min-chars 600

Dependencies
------------
    pip install chardet beautifulsoup4 lxml ftfy
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from text_cleaners import (              # noqa: E402
    ALPHA_TOKEN_MIN_FRACTION,
    MAX_NONASCII_FRACTION,
    WORDS_IN_A_ROW_THRESHOLD,
    WORD_RE,
    has_natural_language_run,
    is_garbled,
    normalize_and_clean,
    read_text_with_detection,
    strip_html,
)

EXTS = {".txt", ".html", ".htm"}


def list_input_files(input_dir: Path) -> List[Path]:
    """Recursively list files under ``input_dir`` with a supported extension.

    Args:
        input_dir: Directory to search recursively.

    Returns:
        A sorted list of files whose suffix is one of ``EXTS``
        (``.txt``/``.html``/``.htm``).
    """
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def relative_output_path(src: Path, input_dir: Path, output_dir: Path) -> Path:
    """Output mirrors the input layout under `output_dir`, with .txt
       extension regardless of input extension."""
    rel = src.relative_to(input_dir)
    return output_dir / rel.with_suffix(".txt")


def filter_corpus(
    input_dir: Path,
    output_dir: Path,
    reject_dir: Path,
    log_csv: Path,
    *,
    dry_run: bool,
    copy_rejects: bool,
    window_size: int,
    alpha_min: float,
    max_nonascii: float,
    garbled_min_chars: int,
    garbled_alpha_frac: float,
) -> dict:
    """Run the keep/reject quality gate over every input file.

    For each file: reads with encoding detection, strips HTML, computes an
    aggressively-normalized form for judging, and keeps it only if it has a
    natural-language run and is not garbled. Kept files are written (as
    ``.txt`` mirroring the input layout) to ``output_dir``; rejected originals
    are moved (or copied) to ``reject_dir``. Writes a CSV decision log unless
    ``dry_run``.

    Args:
        input_dir: Root directory of raw input files.
        output_dir: Where kept files are written.
        reject_dir: Where rejected originals are moved/copied.
        log_csv: Path for the decision-log CSV.
        dry_run: If True, judge and log only; write/move nothing.
        copy_rejects: If True, copy rejected originals instead of moving them.
        window_size: Sliding-window size (words) for NL detection.
        alpha_min: Minimum alphabetic-token fraction per window.
        max_nonascii: Maximum average per-token non-ASCII fraction per window.
        garbled_min_chars: Minimum cleaned length before a file can be kept.
        garbled_alpha_frac: Minimum alpha/non-space ratio before a file can be kept.

    Returns:
        A stats dict with ``total_files``, ``kept``, ``rejected`` and
        ``errors`` counts.
    """
    files = list_input_files(input_dir)
    stats = {"total_files": len(files), "kept": 0, "rejected": 0, "errors": 0}
    log_rows = []

    for src in files:
        try:
            raw = read_text_with_detection(src)
            stripped = strip_html(raw, src.suffix.lower())

            # `normalized` is the aggressively-cleaned form (no punctuation,
            # single line) used ONLY for the keep/reject decision. The text
            # we actually write to disk is `stripped` — encoding-fixed and
            # HTML-stripped but with paragraph structure intact, so downstream
            # tools like clean_web_corpus.py and corpus_to_hybrid_db.py get
            # something useful. (Original preprocessing/corpus_filtering.py
            # wrote `normalized`; that produced unreadable output.)
            normalized = normalize_and_clean(stripped)

            ok = has_natural_language_run(
                normalized,
                window_size=window_size,
                alpha_min_frac=alpha_min,
                max_nonascii_frac=max_nonascii,
            )
            junky = is_garbled(
                normalized,
                min_chars=garbled_min_chars,
                alpha_min_frac=garbled_alpha_frac,
            )
            decision = "keep" if (ok and not junky) else "reject"

            if decision == "keep":
                dst = relative_output_path(src, input_dir, output_dir)
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst, "w", encoding="utf-8", newline="\n") as f:
                        f.write(stripped)
                stats["kept"] += 1
                print(f"[KEEP] {src.relative_to(input_dir)}  "
                      f"({len(stripped)} chars written, "
                      f"{len(normalized)} chars judged)")
            else:
                rej = reject_dir / src.relative_to(input_dir)
                if not dry_run:
                    rej.parent.mkdir(parents=True, exist_ok=True)
                    if copy_rejects:
                        shutil.copy2(str(src), str(rej))
                    else:
                        shutil.move(str(src), str(rej))
                stats["rejected"] += 1
                print(f"[REJECT] {src.relative_to(input_dir)}  "
                      f"({len(normalized)} chars judged)")

            log_rows.append({
                "source_path": str(src),
                "decision":    decision,
                "clean_chars": len(normalized),
                "total_words": len(WORD_RE.findall(normalized)),
            })

        except Exception as e:
            stats["errors"] += 1
            log_rows.append({
                "source_path": str(src),
                "decision":    f"error: {type(e).__name__}: {e}",
                "clean_chars": 0,
                "total_words": 0,
            })
            print(f"[ERROR] {src}: {type(e).__name__}: {e}", file=sys.stderr)

    if not dry_run:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["source_path", "decision", "clean_chars", "total_words"]
            )
            w.writeheader()
            w.writerows(log_rows)

    return stats


def main(argv=None) -> int:
    """Parse args and run the file-level corpus quality gate.

    Resolves input/output/reject directories and the log path, then delegates
    to ``filter_corpus`` and prints a summary.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when None.

    Returns:
        Process exit code: 0 on success, 2 if the input directory is missing.
    """
    parser = argparse.ArgumentParser(
        description="File-level quality gate over a raw text corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory of raw .txt/.html files (recurses).")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to write kept files (.txt, mirrors input layout).")
    parser.add_argument("--reject-dir", required=True, type=Path,
                        help="Where to move rejected originals.")
    parser.add_argument("--log", type=Path, default=None,
                        help="Path to decision CSV. Default: <output-dir>/clean_log.csv.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report decisions, write nothing, move nothing.")
    parser.add_argument("--copy-rejects", action="store_true",
                        help="Copy rejected originals into --reject-dir instead of moving them.")
    parser.add_argument("--window-size", type=int, default=WORDS_IN_A_ROW_THRESHOLD,
                        help=f"Sliding-window size in words for NL detection. "
                             f"Default: {WORDS_IN_A_ROW_THRESHOLD}.")
    parser.add_argument("--alpha-min", type=float, default=ALPHA_TOKEN_MIN_FRACTION,
                        help=f"Min alphabetic-token fraction per window. "
                             f"Default: {ALPHA_TOKEN_MIN_FRACTION}.")
    parser.add_argument("--max-nonascii", type=float, default=MAX_NONASCII_FRACTION,
                        help=f"Max average per-token non-ASCII fraction per window. "
                             f"Default: {MAX_NONASCII_FRACTION}.")
    parser.add_argument("--garbled-min-chars", type=int, default=400,
                        help="Reject if cleaned text shorter than this. Default 400.")
    parser.add_argument("--garbled-alpha-frac", type=float, default=0.55,
                        help="Reject if alpha/non-space ratio below this. Default 0.55.")
    args = parser.parse_args(argv)

    input_dir  = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reject_dir = args.reject_dir.expanduser().resolve()
    log_csv    = (args.log or (output_dir / "clean_log.csv")).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    stats = filter_corpus(
        input_dir=input_dir,
        output_dir=output_dir,
        reject_dir=reject_dir,
        log_csv=log_csv,
        dry_run=args.dry_run,
        copy_rejects=args.copy_rejects,
        window_size=args.window_size,
        alpha_min=args.alpha_min,
        max_nonascii=args.max_nonascii,
        garbled_min_chars=args.garbled_min_chars,
        garbled_alpha_frac=args.garbled_alpha_frac,
    )

    print()
    print("=" * 60)
    print(" Summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:14s}: {v}")
    if not args.dry_run:
        print(f"  log           : {log_csv}")
        print(f"  kept under    : {output_dir}")
        print(f"  rejected under: {reject_dir}")
    else:
        print("  (DRY-RUN: nothing written or moved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
