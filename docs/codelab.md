# Codelab — End-to-End on a Tiny Corpus

This walkthrough runs the entire pipeline against a small Substack
publication so every stage finishes in seconds-to-minutes. No GPU is
required. By the end you will have crawled real articles, cleaned them,
ingested them into a fresh SQLite DB, queried them directly, and
optionally served them via the Flask API.

> **What we skip:** training (stage 4). Producing a LoRA adapter takes
> hours on a GPU. The codelab uses an existing endpoint or the
> simulator adapter, which is enough to exercise the retrieval pipeline.

## What you'll need

- macOS or Linux with **Python 3.11** (the venv at `venv311/` is what
  the team uses).
- ~2 GB free disk.
- The seedsoftruth checkout (this repo).
- Internet for the crawl + (optionally) for the HF endpoint.

Estimated time: **20–40 minutes** end-to-end.

## Setup

From the seedsoftruth checkout root:

```bash
# Activate the venv used by the rest of the project.
source venv311/bin/activate     # or venv/bin/activate

# Quick sanity check — these should import without error.
python3 -c "import spacy, sqlite3, requests; print('ok')"
python3 -c "import spacy; spacy.load('en_core_web_sm'); print('spacy model ok')"

# If the spaCy model isn't installed:
python3 -m spacy download en_core_web_sm

# Playwright needs Chromium on disk for the crawler:
python3 -m playwright install chromium
```

We'll use a scratch directory for everything so we don't touch
production data:

```bash
mkdir -p /tmp/codelab/{corpus_raw,corpus_clean}
SCRATCH_DB=/tmp/codelab/mini.db
```

## Stage 1 — Crawl a small Substack

We'll pull just five posts from `peerservice.substack.com` (a public,
small publication). The wrapper enumerates via the Substack JSON API
and feeds article URLs through the crawler.

```bash
# Generate the URL list (the wrapper has a --dry-run mode).
cd webscraper
./substack_scrape.sh --dry-run peerservice.substack.com | head -5 > /tmp/codelab/seeds.txt
cat /tmp/codelab/seeds.txt
```

Output:

```
https://peerservice.substack.com/p/cultivating-seeds-of-truth
https://peerservice.substack.com/p/ufo-disclosure-explained-new-solutions
https://peerservice.substack.com/p/epstein-files-pt-2-beyond-sex-traffickingzorro
https://peerservice.substack.com/p/epstein-files-pt-1-what-official
https://peerservice.substack.com/p/turning-30-reflections-on-transforming
```

Now crawl each URL one at a time (no link traversal):

```bash
while IFS= read -r url; do
  python3 ./web_scraper.py \
    --start-url "$url" \
    --output-dir /tmp/codelab/corpus_raw \
    --cache-db /tmp/codelab/crawl_cache.db \
    --only-root --max-depth 0
done < /tmp/codelab/seeds.txt
```

Check the output:

```bash
ls /tmp/codelab/corpus_raw/
# peerservice.substack.com_p_cultivating-seeds-of-truth.html
# peerservice.substack.com_p_cultivating-seeds-of-truth.txt
# ... (10 files total — html + txt for each of 5 posts)

wc -l /tmp/codelab/corpus_raw/*.txt
# Each .txt is a few hundred to a few thousand lines.
```

Inspect one:

```bash
head -10 /tmp/codelab/corpus_raw/peerservice.substack.com_p_cultivating-seeds-of-truth.txt
```

You should see the article title and the first paragraphs of the body.

### What just happened

`web_scraper.py` for each URL:
1. HEAD-probed for content type → got `text/html`.
2. Rendered the page in headless Chromium (Playwright).
3. Ran `newspaper3k` to extract the article body from the page chrome.
4. Wrote both raw HTML and extracted TXT.

Each URL took a few seconds because Playwright has startup cost; this
is normal.

## Stage 2 — Clean the corpus

```bash
cd ..    # back to seedsoftruth root

# Dry-run first to see what would happen.
python3 tools/clean/clean_web_corpus.py \
  --input-dir /tmp/codelab/corpus_raw \
  --output-dir /tmp/codelab/corpus_clean \
  --dry-run
```

Output (abbreviated):

```
[DRY-RUN] peerservice.substack.com_p_cultivating-seeds-of-truth.txt → ...
          retained: 68.0% (5099/7504 chars) | footer_cut: 989 chars
[DRY-RUN] peerservice.substack.com_p_ufo-disclosure-explained-new-solutions.txt → ...
          retained: 65.3% | footer_cut: 989 chars
...
Summary — 5 input file(s)
  kept                 : 5
  dropped              : 0
```

Two things to notice:

- **~30% of each file got dropped.** The cleaner removed the repeated
  PEERS/WTK footer (`footer_cut: 989 chars`) and below-threshold
  short blocks.
- **Nothing got rejected as a listing page or JS stub.** The
  per-article URLs we crawled are real articles, not chrome.

Now run for real:

```bash
python3 tools/clean/clean_web_corpus.py \
  --input-dir /tmp/codelab/corpus_raw \
  --output-dir /tmp/codelab/corpus_clean

ls /tmp/codelab/corpus_clean/
# peerservice.substack.com_p_cultivating-seeds-of-truth_cleaned.txt
# ... (5 cleaned files)
```

Compare a raw and a cleaned file to see what changed:

```bash
diff <(tail -20 /tmp/codelab/corpus_raw/peerservice.substack.com_p_cultivating-seeds-of-truth.txt) \
     <(tail -20 /tmp/codelab/corpus_clean/peerservice.substack.com_p_cultivating-seeds-of-truth_cleaned.txt)
```

You'll see the cleaned tail ends mid-article body; the raw tail has the
"WantToKnow.info is a nonprofit news information service founded by..."
closer that the cleaner cut.

## Stage 3 — Ingest into a fresh DB

We'll build a tiny standalone DB (not append to the production one).

```bash
# Build a URL-map CSV — lets the chunks remember their source URL.
{
  echo "filename,source_url"
  for f in /tmp/codelab/corpus_clean/*_cleaned.txt; do
    name=$(basename "$f")
    slug=$(echo "$name" | sed -E 's/^peerservice\.substack\.com_p_(.+)_cleaned\.txt$/\1/')
    echo "${name},https://peerservice.substack.com/p/${slug}"
  done
} > /tmp/codelab/url_map.csv
cat /tmp/codelab/url_map.csv
```

Now ingest. We'll skip topic classification (it requires a GPU and
external API) but keep NER:

```bash
python3 tools/index/corpus_to_hybrid_db.py ingest \
  --corpus /tmp/codelab/corpus_clean \
  --db    /tmp/codelab/mini.db \
  --subset-name "Codelab Demo" \
  --domain peerservice.substack.com \
  --url-map /tmp/codelab/url_map.csv
```

Watch the progress bar. For 5 articles this finishes in under a minute.
Verify:

```bash
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("/tmp/codelab/mini.db")
print("chunks   :", c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
print("entities :", c.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
print("subsets  :", c.execute("SELECT DISTINCT subset_name FROM chunks").fetchall())
print("sample   :", c.execute("SELECT title, length(fulltext_text), source_url "
                              "FROM chunks LIMIT 3").fetchall())
PY
```

Expected output:

```
chunks   : ~25
entities : ~120  (depends on the articles)
subsets  : [('Codelab Demo',)]
sample   : [('peerservice.substack.com_p_cultivating-seeds-of-truth_cleaned', 1234, 'https://peerservice...'), ...]
```

## Stage 5a — Query directly (no Flask)

The simplest way to use the DB is to call `rag_controller` directly
from Python. This is also what the test suite does.

Create `/tmp/codelab/query.py`:

```python
import asyncio, os, sys
os.environ["HYBRID_DB_PATH"] = "/tmp/codelab/mini.db"
sys.path.insert(0, "/Volumes/SSK/seedsoftruth")   # repo root

import rag_controller as rc

state = rc.boot()

queries = [
    "What is Seeds of Truth?",
    "Cultivating seeds of truth in AI",
    "epstein ranch scientific agenda",
    "cadmium schooner swallow",   # off-corpus — should decline
]
for q in queries:
    out = asyncio.run(rc.search_references(state, q, top_k=3))
    print(f"\n--- {q!r}")
    print(f"   gate={out.get('gate_decision')}/{out.get('gate_reason')}  "
          f"branch={out.get('fts_branch_used')}  top1={out.get('top1_score')}")
    for r in out["results"]:
        print(f"   {r.get('score_bm25'):>6.2f}  {r.get('title')!r}")
        print(f"          {r.get('source_url')}")
```

Run it:

```bash
python3 /tmp/codelab/query.py 2>&1 | grep -vE "^(2026|Creating|llm_model|Booting|Boot:)"
```

Expected behavior:

- The first three queries return 1–3 chunks each, all from your tiny
  ingested corpus.
- The fourth query (`cadmium schooner swallow`) returns 0 results with
  `gate=decline/score_below_floor` — the min-gate is doing its job.

This single Python file is a complete RAG-search application against
your tiny DB. To turn it into an LLM-driven QA app, you'd add a call
to `model_adapters.SimEndpointLLM` (returns the prompt back), or
configure HF/DeepInfra credentials to call a real model.

## Stage 5b — Serve via Flask (optional)

To run the same queries through the real Flask app and UI:

```bash
# Point the server at the codelab DB.
export HYBRID_DB_PATH=/tmp/codelab/mini.db

# Run Flask. (Adjust port / workers as needed.)
gunicorn -c chat_server/gunicorn.conf.py app:app
```

Watch the boot log — you want to see:

```
INFO | rag | Boot: loaded hybrid DB from /tmp/codelab/mini.db
INFO | rag | Boot: loaded flat lookup entries=303,621 from ...
```

In another terminal:

```bash
# Hit /api/search
curl -s -X POST http://localhost:5000/api/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "cultivating seeds of truth",
    "top_k": 5,
    "shard_k": 20,
    "subsets": ["Codelab Demo"],
    "rag_algo_type": 5
  }' | python3 -m json.tool | head -40
```

Expected: at least one result with `"subset": "Codelab Demo"`.

To use the UI: open `http://localhost:5000` in a browser. You won't
see "Codelab Demo" in the subset combos because the UI only lists
combos defined in `static/app.js`. To wire it in:

1. Edit `static/app.js` and add a combo:

   ```js
   const SUBSET_COMBOS = [
     { combo_name: "Codelab", subsets: ["Codelab Demo"] },
     ...
   ];
   ```

2. Add the option to `templates/index.html`:

   ```html
   <option value="Codelab">Codelab</option>
   ```

3. Refresh the browser (no server restart needed for these
   client-only changes).

## Stage 6 — Cleanup

```bash
# Stop Flask (Ctrl-C in its terminal)
# Remove the scratch directory
rm -rf /tmp/codelab/
```

The production DB, your venv, and any non-codelab data are untouched.

## What you just exercised

| Stage | What you did | What it produced |
|---|---|---|
| 1. Crawl | `web_scraper.py` × 5 URLs | 10 files in `corpus_raw/` |
| 2. Clean | `clean_web_corpus.py` | 5 files in `corpus_clean/` |
| 3. Ingest | `corpus_to_hybrid_db.py ingest` | `mini.db` with ~25 chunks |
| 4. Train | *skipped* | *would have produced a LoRA adapter* |
| 5. Serve | Direct `rag_controller` calls; optional Flask | Retrieval results |
| 6. Use | Python / curl / browser | Answers to queries |

This is the full developer flow. The only stages that scale up
non-trivially are the training (stage 4 — GPU hours) and the corpus
size (stage 3 — minutes per thousand articles). Everything else looks
roughly the same whether you have 5 articles or 50,000.

## Next steps

- Add 5 more URLs to the seed file and re-ingest into the same DB.
  Re-running the ingest is idempotent (SHA1 dedupe), so existing chunks
  are skipped.
- Experiment with `rag_algo_type=4` vs `5` to see the V5 boost
  contribution: `out4 = search_references(..., rag_algo_choice=4)`
  vs `out5 = search_references(..., rag_algo_choice=5)`. The
  `topic_boost` and `topic_score` fields tell you where the boost
  fired.
- Read [03-ingest.md](03-ingest.md) for the chunking knobs
  (`--chunk-words`, `--overlap-words`) and try a smaller chunk size to
  see how it affects retrieval precision/recall.
- Run the test suite: `python3 tests/test_retrieval_gate.py`. It
  builds its own ephemeral DBs and exercises every layer of the
  retrieval pipeline.

If you got through this codelab cleanly, you understand the system. The
stage docs are now reference material — read them when a specific
question comes up, not cover-to-cover.
