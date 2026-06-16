# Stage 3 — Ingest into the Hybrid SQLite DB

This stage takes the directory of cleaned `.txt` files from stage 2 and
builds the **corpus DB** — the single SQLite file that the running
service queries at every request. The tool is
`tools/index/corpus_to_hybrid_db.py`.

## What's in the DB

The whole retrieval system reads from one SQLite file:
`data/gamma_master_hybrid_fts_stage3.db` (~62k chunks across
~19 source publications). Its schema:

```sql
chunks(
  lookup_id      INTEGER PRIMARY KEY,  -- FTS5 rowid joins on this
  chunk_id       TEXT,                  -- SHA-style content hash (dedupe key)
  title          TEXT,                  -- per-document title
  subset_name    TEXT,                  -- e.g. 'WTK Archive', 'PEERS Substack'
  domain         TEXT,                  -- e.g. 'peerservice.substack.com'
  source_url     TEXT,                  -- link back to original article
  entities_text  TEXT,                  -- "Jeffrey Epstein, JPMorgan, ..."
  fulltext_text  TEXT                   -- the chunk text itself
)

entities(
  entity_id      INTEGER PK AUTOINCREMENT,
  canonical_name TEXT,
  type           TEXT,                  -- PERSON / ORG / GPE / etc.
  UNIQUE(canonical_name, type)
)

chunk_entities(chunk_lookup_id, entity_id, PRIMARY KEY both, FK both)

topics(
  topic_id    INTEGER PK AUTOINCREMENT,
  domain      TEXT,                     -- e.g. 'health_science'
  topic_name  TEXT,                     -- e.g. 'vaccines_and_immunization'
  UNIQUE(domain, topic_name)
)

chunk_topics(chunk_lookup_id, topic_id, PRIMARY KEY both, FK both)

-- FTS5 indexes — these are what the retriever actually queries
entities_fts  USING fts5(entities_text, tokenize='unicode61')
fulltext_fts  USING fts5(title, fulltext_text, tokenize='unicode61')
```

Two important conventions:

- **FTS5 rowid == `chunks.lookup_id`**. The chunker inserts into both
  tables with the same rowid so we can `JOIN ... ON lookup_id =
  fulltext_fts.rowid` cheaply.
- **`subset_name` is a free-text tag, not a foreign key.** The UI
  filters by passing a list of `subset_name` values. Adding a new
  publication means picking a stable string (like `"PEERS Substack"`)
  and tagging every chunk from that source with it.

There's no embedding column — see the [overview's "How retrieval works"
section](00-overview.md#how-retrieval-works-bm25-not-embeddings) for why.

## The tool

### `tools/index/corpus_to_hybrid_db.py`

Two subcommands:

**`ingest`** — chunk a directory of `.txt` files into a new or existing
DB. This is what you run after stage 2.

```bash
python3 tools/index/corpus_to_hybrid_db.py ingest \
  --corpus webscraper/corpus1_cleaned_peers_substack \
  --db data/gamma_master_hybrid_fts_stage3.db \
  --subset-name "PEERS Substack" \
  --domain peerservice.substack.com \
  --url-map webscraper/peers_substack_url_map.csv
```

**`enrich`** — re-run NER and/or topic classification over chunks
that are already in the DB. Useful when you ingested fast (`--no-ner`)
and want to fill in entities/topics later.

```bash
python3 tools/index/corpus_to_hybrid_db.py enrich \
  --db data/gamma_master_hybrid_fts_stage3.db \
  --entities --topics --rebuild
```

### Key flags

For `ingest`:

```
--corpus DIR              The cleaned-txt directory from stage 2.
--db PATH                 Output SQLite file. Append-by-default; existing
                          rows aren't touched (SHA1 dedupe).
--subset-name STR         Required. Tagged on every chunk this run
                          inserts. Show up in the UI's subset combos.
--domain STR              Default per-chunk domain (the topic classifier
                          overrides this when --with-topics is set).
--url-map CSV             Optional CSV mapping filename → source_url so
                          each chunk inherits its article's URL.
                          (Without this, source_url is empty.)
--recursive               Recurse into subdirectories.
--chunk-words N           Target chunk size in words (default ~250).
--overlap-words N         Word overlap between consecutive chunks
                          (default ~50, for context preservation).
--min-words N             Minimum words to emit a chunk.
--no-dedupe               Disable sha1-of-text dedupe.
--no-clean                Skip the sentence-level clean_chunks.py filter.
--no-ner                  Skip spaCy NER (chunks + fulltext_fts only).
--with-topics             Run BGE + LLM topic classifier (slow, GPU-only).
--clean-existing          Delete the DB file first (rare; usually you
                          want append).
```

### URL-map CSV format

A simple two-column header CSV:

```
filename,source_url
peerservice.substack.com_p_cultivating-seeds-of-truth_cleaned.txt,https://peerservice.substack.com/p/cultivating-seeds-of-truth
peerservice.substack.com_p_a-metaphor-for-bridging-the-deep_cleaned.txt,https://peerservice.substack.com/p/a-metaphor-for-bridging-the-deep
...
```

The filename column matches `Path(path).name` of each file in
`--corpus`. Each chunk inherits the `source_url` of its parent file, so
the UI can link result snippets back to the live article. **Always
include a URL map for web-sourced corpora** — without it, results
display without "View source" links.

## What happens during ingest

For each `.txt` file:

1. **Read and lightly clean** with `clean_chunks.py` (unless
   `--no-clean`). This is a sentence-level pass that drops navigation
   chrome / CTA lines that survived stage 2.
2. **Chunk** into overlapping word-windows of `--chunk-words` size with
   `--overlap-words` overlap. The chunker tries to avoid breaking
   mid-sentence by walking a bit if the boundary falls inside a quote
   or a word.
3. **Dedupe** by SHA-1 of the chunk text. Identical chunks from
   different files (e.g., shared boilerplate) get inserted once.
4. **Run NER** (spaCy `en_core_web_sm`) over the chunk if `--no-ner`
   isn't set. Each named entity (PERSON, ORG, GPE, EVENT, WORK_OF_ART,
   DATE) is upserted into `entities` and linked in `chunk_entities`.
5. **Optionally classify topics** (BGE embedding + LLM, only with
   `--with-topics`). Each chunk gets 0–5 topic tags via
   `chunk_topics`.
6. **Insert into `chunks`** with `lookup_id` set explicitly, then into
   `fulltext_fts` (with same rowid) and `entities_fts`.

A progress bar shows files processed; the final summary reports
chunks inserted, dedup hits, and any failures.

## Sanity-checking after ingest

After ingest, count rows per subset:

```python
import sqlite3
c = sqlite3.connect("data/gamma_master_hybrid_fts_stage3.db")
for s, n in c.execute(
    "SELECT subset_name, COUNT(*) FROM chunks GROUP BY subset_name ORDER BY 2 DESC"
):
    print(f"{n:>6}  {s!r}")
```

If a brand-new subset has 0 chunks, something went wrong — usually
either `--subset-name` was mistyped or the files in `--corpus` were
all empty after cleaning. Check the cleaner's dry-run output.

Probe FTS5 to make sure your new chunks are findable:

```python
c.execute(
    "SELECT COUNT(*) FROM fulltext_fts WHERE fulltext_fts MATCH ?",
    ('"some unique phrase from a new article"',)
).fetchone()
```

Should return at least 1. If it returns 0, the article's text didn't
make it through the cleaner intact — diff
`webscraper/corpus1_cleaned/<file>.txt` against the chunk in
`chunks.fulltext_text`.

## Updating an existing DB

Adding a new subset is non-destructive. Run `ingest` again, pointing at
your new cleaned-text dir with a fresh `--subset-name`. Existing rows
under other subsets are not touched. Re-running with the same subset
name + identical text just no-ops (SHA-1 dedupe).

To delete a subset and re-ingest it:

```sql
DELETE FROM chunk_entities WHERE chunk_lookup_id IN
  (SELECT lookup_id FROM chunks WHERE subset_name = 'PEERS Substack');
DELETE FROM chunk_topics   WHERE chunk_lookup_id IN
  (SELECT lookup_id FROM chunks WHERE subset_name = 'PEERS Substack');
-- delete from FTS5 by deleting from chunks (FTS5 with content=chunks would
-- cascade — but our schema is non-content; FTS rows need explicit cleanup)
DELETE FROM fulltext_fts WHERE rowid IN
  (SELECT lookup_id FROM chunks WHERE subset_name = 'PEERS Substack');
DELETE FROM entities_fts WHERE rowid IN
  (SELECT lookup_id FROM chunks WHERE subset_name = 'PEERS Substack');
DELETE FROM chunks WHERE subset_name = 'PEERS Substack';
```

Always back up the DB before destructive ops:

```bash
cp data/gamma_master_hybrid_fts_stage3.db \
   data/gamma_master_hybrid_fts_stage3.db.bak.$(date +%Y%m%d_%H%M%S)
```

The `.bak.<timestamp>` naming matches what's already in `data/`.

## How retrieval uses what we just built

The retriever (next stage) hits two FTS5 indexes in parallel:

- `entities_fts` — given the entity terms extracted from the user's
  query, find chunks whose pre-extracted entities overlap.
- `fulltext_fts` — given the query text, find chunks whose title +
  body match. Tries phrase first, then strict-AND, then broad-OR.

The two result sets are merged by `lookup_id` and re-ranked. See
[05-serving.md](05-serving.md) for the algorithms.

## Adding the new subset to the UI

After a new subset_name is ingested, you need to add it to the UI's
**subset combos** in `static/app.js`. Find the `SUBSET_COMBOS` array
and add your new subset name to every combo where you want it to
appear. For example, to add `"PEERS Substack"` everywhere WTK appears:

```js
const SUBSET_COMBOS = [
  { combo_name: "WantToKnow",
    subsets: ["WTK Archive", "WantToKnow.info", "PEERS Substack"] },
  { combo_name: "Deep Politics",
    subsets: ["WTK Archive", "WantToKnow.info", "PEERS Substack", ...] },
  ...
];
```

The combo names (left side) appear in the `<select>` in
`templates/index.html`; the subset arrays (right side) become the
`subsets: [...]` payload sent to `/api/search`.

## Where to go next

Once the DB has your new content, you have two paths:

- To use it right now via RAG only (no model retrain), restart the
  Flask app and go to [05-serving.md](05-serving.md).
- To eventually retrain the LLM on the expanded corpus, see
  [04-training.md](04-training.md).
