#!/usr/bin/env bash
#
# mineru_batch.sh
# ================
#
# Batch convert every .pdf under INPUT_DIR to plain text using the
# MinerU CLI, preserving relative directory structure under OUTPUT_DIR.
#
# This sits between `epub_to_pdf_batch.py` (if you started with EPUBs)
# and the python text-cleaners. Failures are logged to
# `$OUTPUT_DIR/mineru_failures.log`.
#
# Prerequisites
# -------------
# Install MinerU (Python package):
#     pip install mineru
# Verify it's on PATH:
#     mineru --help
#
# Usage
# -----
#     ./mineru_batch.sh <input_dir> <output_dir>
#
# Output layout
# -------------
# For each input `INPUT_DIR/sub/a.pdf`, MinerU writes its results
# under `OUTPUT_DIR/sub/a/` (it produces multiple files per PDF;
# look in there for the .md/.txt output).
#
# Notes
# -----
# Uses `--method txt -b pipeline` (text-extraction backend). For
# scanned PDFs you'll want a different MinerU mode; consult the
# MinerU docs.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/mineru_failures.log"
: > "$LOG_FILE"

echo "======================================================"
echo " MinerU Batch Conversion"
echo " Input:  $INPUT_DIR"
echo " Output: $OUTPUT_DIR"
echo "======================================================"
echo

mapfile -t pdfs < <(find "$INPUT_DIR" -type f -iname "*.pdf")
echo "Found ${#pdfs[@]} PDFs."
echo

for pdf in "${pdfs[@]}"; do
    rel_path="${pdf#$INPUT_DIR/}"
    base_name="$(basename "$pdf")"
    name="${base_name%.*}"

    out_dir="$OUTPUT_DIR/$(dirname "$rel_path")/$name"
    mkdir -p "$out_dir"

    echo "Processing: $rel_path"
    echo "  mineru -p \"$pdf\" -o \"$out_dir\" --method txt -b pipeline"

    if mineru \
        -p "$pdf" \
        -o "$out_dir" \
        --method txt \
        -b pipeline
    then
        echo "  ok: $rel_path"
    else
        echo "  FAILED: $rel_path"
        echo "$rel_path" >> "$LOG_FILE"
    fi
    echo
done

echo "======================================================"
echo "All done!"
echo "Failures logged to: $LOG_FILE"
echo "Converted output under: $OUTPUT_DIR"
echo "======================================================"
