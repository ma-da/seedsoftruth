#!/bin/bash
# Bootstrap eval data: clean stale judge-cache entries, then (re-)run the
# LLM labeler against eval/queries_seed.jsonl to produce eval/labels.jsonl.
#
# Run from the project root:
#   ./scripts/bootstrap_eval.sh
#
# Run this when you:
#   - add new queries to eval/queries_seed.jsonl
#   - want to re-judge after a transient labeler failure (e.g. 401 errors
#     left grade=null rows in eval/judge_cache.jsonl)
#
# For the actual A/B comparison after this finishes, run scripts/run_eval.sh.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:5000}"
EVAL_DIR="eval"

# -------- [1] Drop null-grade rows from the judge cache --------
# Forces failed (query, doc) pairs to be re-judged on the next labeler run.
python3 -c "
import json, pathlib
p = pathlib.Path('${EVAL_DIR}/judge_cache.jsonl')
keep = [json.loads(l) for l in p.read_text().splitlines() if l.strip() and json.loads(l).get('grade') is not None]
p.write_text('\n'.join(json.dumps(k) for k in keep) + '\n')
print(f'kept {len(keep)} graded entries')
"

# -------- [2] (Re-)label — cheap if cache is warm; only new pairs hit the LLM --------
: "${DEEPINFRA_API_KEY:?DEEPINFRA_API_KEY not set — required by labeler.py}"
python3 ${EVAL_DIR}/labeler.py \
    --judge deepinfra \
    --queries ${EVAL_DIR}/queries_seed.jsonl \
    --base-url "${BASE_URL}" \
    --output ${EVAL_DIR}/labels.jsonl \
    --cache ${EVAL_DIR}/judge_cache.jsonl
