#!/usr/bin/env python3
"""
epub_to_pdf_batch.py
====================

Batch-convert every .epub in a source directory to .pdf in a target
directory, using Calibre's `ebook-convert` CLI.

This is the first step in the prep pipeline for corpora that arrive
as EPUBs:

    .epub  →  .pdf  (this script)
    .pdf   →  .txt  (mineru_batch.sh)
    .txt   → quality-filtered .txt  (filter_corpus.py)
    .txt   → cleaned .txt  (clean_web_corpus.py)
    .txt   → SQLite hybrid-FTS DB  (corpus_to_hybrid_db.py ingest)

Prerequisites
-------------
Install Calibre (provides `ebook-convert`):

    # macOS
    brew install --cask calibre

    # Ubuntu / Debian
    sudo apt install calibre

Verify with:

    ebook-convert --version

Usage
-----
    python tools/ingest/epub_to_pdf_batch.py SRC_DIR DST_DIR

Layout
------
Output PDFs are placed directly under DST_DIR (no recursion mirror).
Failures print an error and continue with the next file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def convert_epub_to_pdf(src_dir: Path, dst_dir: Path) -> int:
    """Convert every ``.epub`` in ``src_dir`` to a ``.pdf`` in ``dst_dir``.

    Runs Calibre's ``ebook-convert`` per file with letter paper and 36pt
    margins, printing progress and continuing past individual failures.

    Args:
        src_dir: Directory containing the source ``.epub`` files.
        dst_dir: Directory to write the generated ``.pdf`` files (created
            if absent).

    Returns:
        Exit code: 0 if all conversions succeeded (or none were found), 1 if
        any file failed, 2 if ``ebook-convert`` is not on PATH.

    Raises:
        ValueError: If ``src_dir`` does not exist or is not a directory.
    """
    if not src_dir.exists() or not src_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    epub_files = sorted(src_dir.glob("*.epub"))
    if not epub_files:
        print("No EPUB files found.")
        return 0

    failures = 0
    for epub_path in epub_files:
        pdf_path = dst_dir / (epub_path.stem + ".pdf")
        print(f"Converting: {epub_path.name} -> {pdf_path.name}")

        try:
            subprocess.run(
                [
                    "ebook-convert",
                    str(epub_path),
                    str(pdf_path),
                    "--paper-size", "letter",
                    "--margin-left",   "36",
                    "--margin-right",  "36",
                    "--margin-top",    "36",
                    "--margin-bottom", "36",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            print(
                "ERROR: 'ebook-convert' not on PATH. Install Calibre first.",
                file=sys.stderr,
            )
            return 2
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {epub_path.name}")
            print(e.stderr.decode(errors="ignore"), file=sys.stderr)
            failures += 1
        else:
            print(f"  OK:  {pdf_path.name}")

    print()
    print(f"Done. {len(epub_files)} attempted, {failures} failed.")
    return 1 if failures else 0


def main(argv=None) -> int:
    """Parse src/dst args and run the EPUB-to-PDF batch conversion.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when None.

    Returns:
        Process exit code propagated from ``convert_epub_to_pdf``, or 1 on a
        fatal error.
    """
    parser = argparse.ArgumentParser(
        description="Batch convert EPUB files to PDF via Calibre.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("src", type=Path, help="Source directory containing .epub files")
    parser.add_argument("dst", type=Path, help="Target directory for generated .pdf files")
    args = parser.parse_args(argv)

    try:
        return convert_epub_to_pdf(args.src.expanduser().resolve(),
                                   args.dst.expanduser().resolve())
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
