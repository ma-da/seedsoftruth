#!/usr/bin/env python3
"""
clean_transcript.py
===================

Strip `"Speaker N HH:MM:SS"`-style header lines from podcast / video
transcripts, keeping only the dialogue. Handles common shapes:

    Speaker 1 12:34
    Speaker 2 01:23:45
    Alice Jones 0:05.123
    Bob Smith 1 10:42

Patterns NOT matched (kept as content): inline timestamps, sentence
text that happens to end with digits, anything not on its own line.

Usage
-----
    # Single file:
    python tools/clean/clean_transcript.py path/to/raw.txt path/to/clean.txt

    # Whole directory:
    python tools/clean/clean_transcript.py \\
        --input-dir  /data/raw_transcripts \\
        --output-dir /data/clean_transcripts

The output is a single trailing newline-terminated text with at most
one blank line between paragraphs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from text_cleaners import strip_speaker_lines      # noqa: E402


def clean_file(input_path: Path, output_path: Path) -> None:
    """Strip speaker/timestamp lines from one transcript file and write it out.

    Args:
        input_path: Path to the raw transcript text file.
        output_path: Path to write the cleaned text (parent dirs created).

    Raises:
        FileNotFoundError: If ``input_path`` is not an existing file.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Cleaning: {input_path.name} → {output_path.name}")
    raw = input_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = strip_speaker_lines(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    print(f"  [OK]  {len(cleaned):,} chars written.")


def clean_directory(input_dir: Path, output_dir: Path) -> None:
    """Clean every ``.txt`` transcript in a directory into ``output_dir``.

    Args:
        input_dir: Directory of raw ``.txt`` transcripts.
        output_dir: Directory to write cleaned files (created if absent).

    Raises:
        ValueError: If ``input_dir`` is not a directory.
    """
    if not input_dir.is_dir():
        raise ValueError(f"Input is not a directory: {input_dir}")
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print("[WARN] No .txt files found.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in txt_files:
        clean_file(p, output_dir / p.name)


def main(argv=None) -> int:
    """Parse args and run transcript cleaning in single-file or batch mode.

    Args:
        argv: Argument vector to parse; defaults to ``sys.argv`` when None.

    Returns:
        Process exit code: 0 on success, 1 if an error is raised during
        cleaning.
    """
    parser = argparse.ArgumentParser(
        description="Strip speaker + timestamp lines from transcripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Two usage modes: positional pair (single file) OR --input-dir/--output-dir (batch)
    parser.add_argument("input", nargs="?",
                        help="Path to a single raw transcript file (single-file mode).")
    parser.add_argument("output", nargs="?",
                        help="Path to write the cleaned file (single-file mode).")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory of .txt transcripts (batch mode).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write cleaned transcripts (batch mode).")
    args = parser.parse_args(argv)

    try:
        if args.input_dir or args.output_dir:
            if not (args.input_dir and args.output_dir):
                parser.error("--input-dir and --output-dir must be passed together")
            clean_directory(
                Path(args.input_dir).expanduser().resolve(),
                Path(args.output_dir).expanduser().resolve(),
            )
        else:
            if not (args.input and args.output):
                parser.error("Pass either two positional args (in, out) "
                             "or --input-dir/--output-dir.")
            clean_file(
                Path(args.input).expanduser().resolve(),
                Path(args.output).expanduser().resolve(),
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
