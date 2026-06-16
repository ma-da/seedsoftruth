#!/usr/bin/env python3
"""
clean_web_corpus.py
===================

Block-level cleaner for a directory of `.txt` files derived from web
scrapes or PDF→TXT conversions. Writes cleaned versions to a target
directory with `_cleaned` appended to each filename.

What it does (per file)
-----------------------
1.  **File-level rejection** (NEW, opt-out-able): drops JS-required
    Substack shells (e.g. ``substack.com/@author`` profile stubs that
    extracted to one line of CTA copy), drops listing/archive/topic
    pages (which duplicate article content with truncated context),
    and drops files smaller than ``--min-file-chars`` after cleaning.
    Also supports user-supplied filename-pattern skips via
    ``--drop-pattern``.
2.  Cuts everything after `**What you can do:**` (configurable via
    `--cut-marker`).
3.  Cuts the repeated PEERS/WantToKnow article-closer (the
    "WantToKnow.info is a nonprofit news information service founded
    by White House whistleblower Fred Burks..." paragraph that 32 of
    38 corpus1 articles share). Configurable via
    ``--repeated-footer-prefix`` (repeatable) and disable-able with
    ``--no-cut-footer``.
4.  Normalizes markdown:
        `[label](url)` → `label`
        `**bold**`     → `bold`
5.  Strips bare URLs.
6.  Drops promotional / meta lines (e.g. "WantToKnow.info", "PEERS",
    "click here", "Note:", "For more information").
7.  Cleans whitespace, then rejoins wrapped paragraph lines into single
    long-line paragraphs (helps after PDF→TXT).
8.  Splits into paragraph blocks. Optionally drops a percentage of
    blocks from the start and/or end of the document via
    `--strip_pre` / `--strip_post` (each 0–25). Useful for stripping
    front/back matter without page numbers.
9.  Drops TOC-like blocks and book-index-like blocks (heuristic).
10. Separates a references section if found (kept out of the cleaned
    output; not written separately by default).
11. Drops very short blocks and SCREAMING-ALL-CAPS blocks.

Usage
-----
    # Dry run on a directory — reports retention but writes nothing.
    python tools/clean/clean_web_corpus.py \\
        --input-dir  /data/raw_txt \\
        --output-dir /data/cleaned_txt \\
        --dry-run

    # Actual cleaning, dropping the first 5% and last 10% of blocks.
    python tools/clean/clean_web_corpus.py \\
        --input-dir  /data/raw_txt \\
        --output-dir /data/cleaned_txt \\
        --strip_pre 5 \\
        --strip_post 10

    # Restore pre-update behavior (no file-level rejection, keep PEERS footer).
    python tools/clean/clean_web_corpus.py \\
        --input-dir  /data/raw_txt \\
        --output-dir /data/cleaned_txt \\
        --no-default-filters

    # Disable just the listing-page dropper.
    python tools/clean/clean_web_corpus.py \\
        --input-dir  /data/raw_txt \\
        --output-dir /data/cleaned_txt \\
        --no-drop-listing-pages

Notes
-----
* Files ending in `_cleaned.txt` are skipped (so re-running the tool
  on its own output dir is safe).
* Existing output files are NOT overwritten — re-runs will skip them.
* A final summary lists what was kept and what was dropped, grouped
  by rejection reason.
* Use the output directory as input to `corpus_to_hybrid_db.py ingest`.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

# Local sibling-module import (works whether invoked directly or imported)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from text_cleaners import (              # noqa: E402
    CUT_MARKER,
    DEFAULT_REPEATED_FOOTERS,
    clean_text_web_corpus,
    evaluate_file_rejection,
    validate_strip_param,
)


def should_ignore(path: Path) -> bool:
    """Return True if the file is already a cleaned output (``_cleaned.txt``)."""
    return path.name.endswith("_cleaned.txt")


def cleaned_name(path: Path) -> str:
    """Return the output filename for ``path`` with ``_cleaned`` before the suffix."""
    return f"{path.stem}_cleaned{path.suffix}"


def _matches_any_pattern(name: str, patterns: Sequence[str]) -> Optional[str]:
    """Return the first fnmatch glob in ``patterns`` that matches
    ``name``, or None. Used by the --drop-pattern CLI option."""
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return pat
    return None


def clean_file(
    input_path: Path,
    output_dir: Path,
    *,
    dry_run: bool,
    strip_pre: int,
    strip_post: int,
    cut_marker: str,
    min_block_chars: int,
    repeated_footer_prefixes: Sequence[str],
    drop_js_stubs: bool,
    drop_listing_pages: bool,
    min_file_chars: int,
    drop_patterns: Sequence[str],
    stats: Counter,
) -> None:
    """Clean one file. Mutates ``stats`` for the end-of-run summary
    (one of: kept, skipped, rejected_<reason>)."""
    if should_ignore(input_path):
        print(f"[SKIP] Already cleaned: {input_path.name}")
        stats["skipped_already_cleaned"] += 1
        return

    # ---- 1) URL-pattern skip --------------------------------------------- #
    pat = _matches_any_pattern(input_path.name, drop_patterns)
    if pat:
        print(f"[DROP] {input_path.name} → matched --drop-pattern {pat!r}")
        stats[f"dropped_pattern:{pat}"] += 1
        return

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # ---- 2) File-level content rejection (JS stub / listing page) -------- #
    # min_chars is applied separately AFTER cleaning so a borderline file
    # gets a fair shot at meeting the threshold once boilerplate is cut.
    rejected, reason = evaluate_file_rejection(
        raw,
        drop_js_stubs=drop_js_stubs,
        drop_listing_pages=drop_listing_pages,
        min_chars=0,
    )
    if rejected:
        print(f"[DROP] {input_path.name} → {reason}")
        stats[f"dropped_{reason.split(':',1)[0]}"] += 1
        return

    # ---- 3) Block-level cleaning ----------------------------------------- #
    result = clean_text_web_corpus(
        raw,
        strip_pre=strip_pre,
        strip_post=strip_post,
        cut_marker=cut_marker,
        min_block_chars=min_block_chars,
        repeated_footer_prefixes=repeated_footer_prefixes,
    )

    out_name = cleaned_name(input_path)
    out_path = output_dir / out_name

    orig_len = result["stats"]["original_chars"]
    clean_len = result["stats"]["clean_chars"]
    retained = (clean_len / orig_len * 100.0) if orig_len else 0.0
    footer_cut = result["stats"].get("repeated_footer_cut", 0)

    # ---- 4) Post-clean size gate ----------------------------------------- #
    if min_file_chars > 0 and clean_len < min_file_chars:
        print(
            f"[DROP] {input_path.name} → too_short:{clean_len}<{min_file_chars}"
            f" (orig {orig_len}, footer_cut {footer_cut})"
        )
        stats["dropped_too_short"] += 1
        return

    if dry_run:
        print(
            f"[DRY-RUN] {input_path.name} → {out_name} | "
            f"retained: {retained:.1f}% ({clean_len}/{orig_len} chars) | "
            f"blocks: {result['stats']['blocks_before_strip']}→"
            f"{result['stats']['blocks_after_strip']} "
            f"(strip_pre={strip_pre}%, strip_post={strip_post}%)"
            + (f" | footer_cut: {footer_cut} chars" if footer_cut else "")
        )
        stats["kept"] += 1
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[SKIP] Output exists: {out_path}")
        stats["skipped_output_exists"] += 1
        return

    with out_path.open("w", encoding="utf-8") as f:
        f.write(result["clean_text"])
    print(f"[OK] {input_path.name} → {out_path}  (retained {retained:.1f}%)")
    stats["kept"] += 1


def clean_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool,
    strip_pre: int,
    strip_post: int,
    cut_marker: str,
    min_block_chars: int,
    repeated_footer_prefixes: Sequence[str],
    drop_js_stubs: bool,
    drop_listing_pages: bool,
    min_file_chars: int,
    drop_patterns: Sequence[str],
    recursive: bool,
) -> Counter:
    """Clean every .txt under ``input_dir``. Returns the stats Counter
    so the caller can drive a final summary."""
    if not input_dir.is_dir():
        raise ValueError(f"Input is not a directory: {input_dir}")

    pattern = "**/*.txt" if recursive else "*.txt"
    files = sorted(input_dir.glob(pattern))
    if not files:
        print("[WARN] No .txt files found.")
        return Counter()

    stats: Counter = Counter()
    for p in files:
        if should_ignore(p):
            print(f"[SKIP] Already cleaned: {p.name}")
            stats["skipped_already_cleaned"] += 1
            continue
        clean_file(
            p,
            output_dir=output_dir,
            dry_run=dry_run,
            strip_pre=strip_pre,
            strip_post=strip_post,
            cut_marker=cut_marker,
            min_block_chars=min_block_chars,
            repeated_footer_prefixes=repeated_footer_prefixes,
            drop_js_stubs=drop_js_stubs,
            drop_listing_pages=drop_listing_pages,
            min_file_chars=min_file_chars,
            drop_patterns=drop_patterns,
            stats=stats,
        )

    _print_summary(stats, total=len(files))
    return stats


def _print_summary(stats: Counter, *, total: int) -> None:
    """Print kept/dropped/skipped breakdown at the end of a run."""
    print()
    print("=" * 60)
    print(f"Summary — {total} input file(s)")
    print("-" * 60)
    print(f"  kept                 : {stats.get('kept', 0)}")
    dropped_total = sum(v for k, v in stats.items() if k.startswith("dropped_"))
    print(f"  dropped              : {dropped_total}")
    for k in sorted(k for k in stats if k.startswith("dropped_")):
        print(f"      {k[8:]:<28} {stats[k]}")
    skipped_total = sum(v for k, v in stats.items() if k.startswith("skipped_"))
    if skipped_total:
        print(f"  skipped              : {skipped_total}")
        for k in sorted(k for k in stats if k.startswith("skipped_")):
            print(f"      {k[8:]:<28} {stats[k]}")
    print("=" * 60)


def main(argv=None) -> int:
    """Parse args and run the block-level web-corpus cleaner over a directory.

    Resolves input/output dirs, validates strip percentages, reconciles the
    file-level rejection and repeated-footer options (honoring
    ``--no-default-filters`` / ``--no-cut-footer``), then delegates to
    ``clean_directory``.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when None.

    Returns:
        Process exit code (0 on success).

    Raises:
        FileNotFoundError: If the input directory does not exist.
    """
    parser = argparse.ArgumentParser(
        description="Clean a directory of .txt files (block-level cleaner).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing .txt files to clean")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write cleaned .txt files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run cleaning without writing output files")
    parser.add_argument("--strip_pre", type=int, default=0,
                        help="Percentage (0–25) of blocks to strip from the beginning")
    parser.add_argument("--strip_post", type=int, default=0,
                        help="Percentage (0–25) of blocks to strip from the end")
    parser.add_argument("--cut-marker", default=CUT_MARKER,
                        help=f'Delete everything after this exact marker '
                             f'(default: {CUT_MARKER!r}). Pass "" to disable.')
    parser.add_argument("--min-block-chars", type=int, default=220,
                        help="Drop blocks shorter than this (after stripping). Default 220.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories of --input-dir.")

    # ---- file-level rejection (new) -------------------------------------- #
    group = parser.add_argument_group(
        "file-level rejection",
        "Drop whole files that aren't worth keeping for RAG/training: "
        "JS-required Substack shells, listing/archive pages, tiny stubs.",
    )
    group.add_argument("--no-default-filters", action="store_true",
                       help="Turn OFF all the new file-level rejection filters "
                            "(restores pre-update behavior).")
    group.add_argument("--drop-listing-pages", dest="drop_listing_pages",
                       action="store_true", default=True,
                       help="Drop Substack listing/archive/topic pages "
                            "(default ON; they duplicate article content).")
    group.add_argument("--no-drop-listing-pages", dest="drop_listing_pages",
                       action="store_false",
                       help="Keep listing/archive/topic pages.")
    group.add_argument("--drop-js-stubs", dest="drop_js_stubs",
                       action="store_true", default=True,
                       help="Drop JS-required Substack shells whose body is "
                            "one line of CTA copy (default ON).")
    group.add_argument("--no-drop-js-stubs", dest="drop_js_stubs",
                       action="store_false",
                       help="Keep JS-required shells.")
    group.add_argument("--min-file-chars", type=int, default=400,
                       help="Drop files whose cleaned text is shorter than this "
                            "many chars. Default 400. Set 0 to disable.")
    group.add_argument("--drop-pattern", action="append", default=[],
                       metavar="GLOB",
                       help="Drop files whose filename matches this glob "
                            "(fnmatch syntax, e.g. '*_archive*'). Repeatable.")

    # ---- repeated-footer cutter (new) ------------------------------------ #
    fgroup = parser.add_argument_group(
        "repeated-footer cut",
        "Cut the PEERS/WantToKnow closer paragraph that appears at the "
        "bottom of most articles. Specify your own prefixes for other corpora.",
    )
    fgroup.add_argument("--repeated-footer-prefix", action="append",
                        default=None, metavar="PREFIX",
                        help="A text prefix that marks the start of a repeated "
                             "footer to cut. Repeatable. If not specified, "
                             "defaults to: "
                             + " | ".join(repr(p) for p in DEFAULT_REPEATED_FOOTERS))
    fgroup.add_argument("--no-cut-footer", action="store_true",
                        help="Disable the repeated-footer cut entirely.")

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    strip_pre = validate_strip_param(args.strip_pre, "--strip_pre")
    strip_post = validate_strip_param(args.strip_post, "--strip_post")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Resolve filter defaults vs --no-default-filters.
    if args.no_default_filters:
        drop_js_stubs = False
        drop_listing_pages = False
        min_file_chars = 0
        repeated_footer_prefixes: Sequence[str] = ()
    else:
        drop_js_stubs = args.drop_js_stubs
        drop_listing_pages = args.drop_listing_pages
        min_file_chars = args.min_file_chars
        if args.no_cut_footer:
            repeated_footer_prefixes = ()
        elif args.repeated_footer_prefix is not None:
            repeated_footer_prefixes = tuple(args.repeated_footer_prefix)
        else:
            repeated_footer_prefixes = DEFAULT_REPEATED_FOOTERS

    clean_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
        strip_pre=strip_pre,
        strip_post=strip_post,
        cut_marker=args.cut_marker,
        min_block_chars=args.min_block_chars,
        repeated_footer_prefixes=repeated_footer_prefixes,
        drop_js_stubs=drop_js_stubs,
        drop_listing_pages=drop_listing_pages,
        min_file_chars=min_file_chars,
        drop_patterns=args.drop_pattern,
        recursive=args.recursive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
