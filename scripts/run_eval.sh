#!/bin/bash
# A/B eval harness for the Seeds of Truth hybrid RAG variants.
#
# Run from the project root:
#   ./scripts/run_eval.sh
#
# The Flask app must be reachable at $BASE_URL (default http://localhost:5000).
#
# The script has three sections. For "normal eval operations" — i.e. you
# changed an algo and want to re-compare the variants — only [3] is needed;
# [1] is a one-time bootstrap (kept here for reference, commented out by
# default) and [2] is a no-op when queries_seed.jsonl is unchanged and the
# judge cache is warm.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:5000}"
EVAL_DIR="eval"

# -------- [1] One-time bootstrap: drop null-grade rows from the judge cache --------
#
# Only useful if judge_cache.jsonl has rows with grade=null from a transient
# failure (e.g. the original ANTHROPIC_API_KEY 401 incident). Stripping them
# here forces those (query, doc) pairs to be re-judged on the next labeler run.
# Leave commented out for normal operation.
#
# python3 -c "
# import json, pathlib
# p = pathlib.Path('${EVAL_DIR}/judge_cache.jsonl')
# keep = [json.loads(l) for l in p.read_text().splitlines() if l.strip() and json.loads(l).get('grade') is not None]
# p.write_text('\n'.join(json.dumps(k) for k in keep) + '\n')
# print(f'kept {len(keep)} graded entries')
# "

# -------- [2] Incremental relabel — needed when you EDIT queries_seed.jsonl --------
#
# Hits cache for everything already judged, only spends LLM credits on new
# (query, doc) pairs. Skip this block if you haven't touched the seed queries
# or invalidated the cache.
#: "${DEEPINFRA_API_KEY:?DEEPINFRA_API_KEY not set — required by labeler.py}"
#python3 ${EVAL_DIR}/labeler.py \
#    --judge deepinfra \
#    --queries ${EVAL_DIR}/queries_seed.jsonl \
#    --base-url "${BASE_URL}" \
#    --output ${EVAL_DIR}/labels.jsonl \
#    --cache ${EVAL_DIR}/judge_cache.jsonl

# -------- [3] A/B comparison — re-run this after every algo change --------
python3 ${EVAL_DIR}/ab_eval.py \
    --eval-set ${EVAL_DIR}/labels.jsonl \
    --variants 1,2,3 \
    --top-k 20 \
    --base-url "${BASE_URL}" \
    --output-dir ${EVAL_DIR}/runs
