#!/usr/bin/env bash
#
# convert_md_to_txt.sh
# =====================
#
# Recursively find every .md file under a directory and produce a
# sibling .txt alongside each one, using pandoc with `--wrap=none`
# (no hard line breaks) and the `plain` writer (no markdown syntax).
#
# Useful when MinerU produces .md output and you want plain .txt for
# the downstream cleaners / `corpus_to_hybrid_db.py ingest`.
#
# Prerequisites
# -------------
# Install pandoc:
#     # macOS
#     brew install pandoc
#     # Ubuntu / Debian
#     sudo apt install pandoc
#
# Usage
# -----
#     ./convert_md_to_txt.sh <target_dir>
#
# Behavior
# --------
# - Existing .txt siblings are overwritten.
# - .md files are left in place (delete them yourself if you no
#   longer need them).
# - One pandoc invocation per file (no parallelism); pipe through
#   `xargs -P` if you need concurrency.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_dir>"
    exit 1
fi

TARGET_DIR="$1"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: '$TARGET_DIR' is not a directory" >&2
    exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
    echo "Error: pandoc not on PATH. Install it first." >&2
    exit 2
fi

count=0
find "$TARGET_DIR" -name "*.md" -type f | while read -r md; do
    txt="${md%.md}.txt"
    echo "Creating: $txt"
    pandoc "$md" -t plain --wrap=none -o "$txt"
    count=$((count + 1))
done

echo
echo "Done."
