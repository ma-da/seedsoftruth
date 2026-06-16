# Stage 5 — Serving (Flask + Retrieval + LLM)

This is what's running when a user opens the chat box. The Flask app
exposes a handful of HTTP endpoints; each one calls into
`rag_controller.py` for retrieval and `model_adapters.py` for the LLM.

## Boot

`app.py` starts via gunicorn (`chat_server/gunicorn.conf.py`). On boot:

1. **Load the corpus DB.** `rag_controller.boot()` reads
   `data/gamma_master_hybrid_fts_stage3.db` (overridable via
   `HYBRID_DB_PATH` env var). This is a connection-per-thread setup —
   the DB is read-only at request time.
2. **Load the entity alias map.** `data/entity_query_normalization_map.flat.json`
   — a ~300k-entry dict mapping normalized entity strings (`"jeff
   epstein"`) to canonical forms (`"Jeffrey Edward Epstein"`).
3. **Load spaCy.** `en_core_web_sm` for runtime NER on queries.
4. **Wire up LLM adapters.** All four adapter classes are instantiated
   so the runtime can pick one per request.
5. **Boot the queue worker thread.** Long-running chat requests
   queue here; the worker drains them serially against the LLM
   endpoint (HF's free tier dislikes parallel requests).

The boot log line you want to see at startup:

```
INFO | rag | Boot: loaded hybrid DB from data/gamma_master_hybrid_fts_stage3.db
INFO | rag | Boot: loaded flat lookup entries=303,621 from data/entity_query_normalization_map.flat.json
```

If `HYBRID_DB_PATH` points at the wrong DB (or you didn't set it and
the default doesn't match the current production file), retrieval will
silently return results from the wrong snapshot. The default in
`rag_controller.py:101` matches docker-compose.yml.

## Endpoints

The main HTTP surface:

```
POST /api/search          Retrieval-only. Returns top-K chunks.
POST /api/chat            Full RAG: retrieve → build context → LLM → return answer.
POST /api/ab_test         Side-by-side A/B with two model adapters.
GET  /api/status          Health: model_ready, is_unlocked, etc.
POST /api/email_response  Subscribe to "notify me when ready".
GET  /                    The single-page app (templates/index.html).
```

`/api/status` accepts an optional `model_type` payload field
(`"hf" | "deepinfra" | "spark" | "sim"`). When provided, the readiness
probe targets that specific adapter; when omitted, it falls back to the
server's global default (set by the `MODEL_ADAPTER` env var, default
`"hf"`). The frontend always sends the currently-selected model so the
"warming…" chip in the UI reflects the model the user would actually
hit if they clicked send.

All `/api/*` endpoints accept JSON. `rag_algo_type` is the integer
retrieval-algorithm selector (see below); it defaults to 5 (V5) on the
server and the frontend always sends 5 explicitly via
`CFG.DEFAULT_RAG_ALGO_TYPE`.

Request shape for `/api/search`:

```json
{
  "query": "public concerns about geoengineering",
  "top_k": 10,
  "shard_k": 20,
  "subsets": ["WTK Archive", "WantToKnow.info", "PEERS Substack"],
  "rag_algo_type": 5
}
```

Response (abbreviated):

```json
{
  "ok": true,
  "num_results": 5,
  "results": [
    { "row_id": "...", "lookup_id": 166100, "title": "...",
      "subset": "PEERS Substack", "source_url": "https://...",
      "score_bm25": 11.32, "topic_boost": 1.0, "topic_score": 11.32,
      "text": "...", "snippet": "...", "snippet_html": "..." },
    ...
  ],
  "gate_decision": "pass",
  "gate_reason": "pass_phrase_match",
  "top1_score": 11.32,
  "fts_branch_used": "phrase",
  "n_query_topics": 2,
  "topic_inference_source": "spacy",
  "boost_alpha": 0.25
}
```

The `gate_*`, `*_score`, and `topic_*` fields are V4/V5-only — they
let the client and the eval harness see why the gate accepted or
rejected the query.

## The retrieval pipeline

`rag_controller.search_references()` is the entry point. The pipeline:

1. **Tokenize the query.** spaCy NER extracts entity spans; each entity
   is looked up in `state.flat_lookup` to canonicalize aliases.
2. **Dispatch by `rag_algo_choice`.**

   | algo | function | what it does |
   |---|---|---|
   | 0 / unrecognized | `_hybrid_search_sqlite` | Legacy v1 |
   | 1 | `_hybrid_search_sqlite` | Plain hybrid (entity_FTS + fulltext_FTS, merged by lookup_id) |
   | 2 | `_hybrid_search_sqlite2` | v1 + tighter merge formula, late-stage filters |
   | 3 | `_hybrid_search_sqlite3` | v2 + canonical-entity coverage + early-anchor checks |
   | 4 | `_hybrid_search_sqlite4` | v2's results, wrapped by a min-gate that catches off-corpus probes |
   | **5** | `_hybrid_search_sqlite5` | **Production default.** v2 retrieval + topic boost + min-gate. |

3. **Apply the min-gate** (V4+ only).
   The gate examines:
   - The top-1 BM25 score against `MIN_GATE_SCORE_FLOOR` (default 17.0).
   - Which FTS branch fired (`phrase` / `strict_and` / `broad_or`).
   - Whether the query had any non-location canonical entities.

   It declines (returns `[]`) when the query looks off-corpus. Decline
   reasons are reported in `gate_reason`:
   - `score_below_floor` — top-1 BM25 < floor (e.g., `"cadmium
     schooner swallow"` random word-salad).
   - `no_entities_broad_or_only` — no canonical entities and only
     broad-OR matched (e.g., the bare query `"Switzerland"` matched
     incidentally).
   - `no_results` — retrieval came back empty.

   It passes with:
   - `pass` — normal in-corpus hit on strict-AND.
   - `pass_phrase_match` — phrase branch fired (added 2026-05 to
     fix the geoengineering-query regression — phrase BM25 is
     systematically lower than strict-AND on the same chunk, so the
     floor doesn't apply).

4. **(V5 only) Topic boost.** Infer the query's topic_ids two ways:
   - Primary: spaCy lemma-match query content tokens against a cached
     per-DB topic-name lemma index.
   - Fallback: take the most common topic_ids on the top-N initial
     candidates (co-occurrence).
   Boost each candidate's BM25 multiplicatively by `1 + α *
   overlap_frac` where `α = V5_TOPIC_BOOST_ALPHA` (default 0.25). Re-sort
   the candidate pool, return top_k.

5. **Late-stage filters and bonuses.** The candidates run through
   `_keyword_overlap_bonus`, `_exact_phrase_bonus`, and proximity
   checks that further penalize incidental matches and reward chunks
   where the query terms appear close together. These are
   `score`-affecting but don't change the gate decision.

6. **Return.** The result rows are shaped by `_sqlite_row_to_result`
   into the dict format above.

## Building the prompt context

`rag_controller.build_context_v2()` takes the top-K retrieved chunks
and assembles a prompt block like:

```
<doc id="dchunk0001" url="https://peerservice.substack.com/p/..." score="22.31">
[chunk text]
</doc>

<doc id="dchunk0002" url="..." score="19.42">
[chunk text]
</doc>
```

This block goes into the LLM prompt between a **system prompt** (from
`model_prompts.py` — DEEP_REPORTING_V1/V2/V3 or SMOKING_MAN) and the
user's actual question. The system prompt tells the model to:

- Classify the question by evidence level (Established / Contested /
  Anomalies / Low evidence).
- Cite specific docs by ID when making claims.
- Decline gracefully when retrieval is empty rather than fabricate.

## The LLM adapters

`model_adapters.py` defines a small `LLMStrategy` ABC and four concrete
adapters. The currently active adapter is selected at request time via
the `model_type` payload field (default `"hf"`):

```
HFEndpointLLM         model_type = "hf"        ← production
DeepInfraLlamaLLM     model_type = "deepinfra"
SparkCloudflareLLM    model_type = "spark"
SimEndpointLLM        model_type = "sim"       ← test/CI
```

Each adapter exposes the same interface (`ask(prompt) -> str`) so the
caller doesn't have to know which one is running. Per-adapter:

- **HFEndpointLLM** — calls a Hugging Face Inference Endpoint running
  Qwen2.5-7B + the project's LoRA. Reads `HF_ENDPOINT_URL` and
  `HF_TOKEN` from env. Free-tier rate-limited, so the queue worker
  serializes requests.
- **DeepInfraLlamaLLM** — fallback when HF endpoint is down. Uses
  Llama-3-70B via DeepInfra's API. Reads `DEEPINFRA_TOKEN`.
- **SparkCloudflareLLM** — wraps a self-hosted model fronted by
  Cloudflare Access; uses service-token headers. For dev only.
- **SimEndpointLLM** — returns the prompt back as the answer, with no
  network call. Used by tests and by the prober.

When you redeploy a new LoRA (stage 4), you update the HF endpoint
behind `HF_ENDPOINT_URL` and the Flask app picks up the new model on
its next request — no Flask restart needed.

## The queue worker

LLM calls are not made directly on the request thread. Instead,
`/api/chat` enqueues a `QueuedJob` and returns immediately with a
`job_id`. A single background thread (started in `boot()`) drains the
queue serially:

```
client POST /api/chat  →  job_id
client GET /api/job/<id>  ←  status: queued
... (worker runs the LLM call) ...
client GET /api/job/<id>  ←  status: done, answer: "..."
```

Why this design: the HF Inference Endpoint free tier strongly prefers
one request at a time. Parallelism would hit 429s.

**Per-job readiness gating.** Each queued job carries the
`model_type` the client picked. The worker peeks at the head of the
queue, runs `is_model_type_ready(job.model_type)` on that specific
adapter, and — if it's cold — fires `send_warmup_for(job.model_type)`
against the same backend before sleeping. This prevents the
asymmetric failure mode where the worker would gate on the default
model's readiness while dispatching to a different one (e.g., HF up
but Spark down → 524 from Cloudflare; or HF down but Spark up → Spark
job sits forever). The same `send_warmup_for(model_type)` helper is
used by `/api/chat`'s queue-then-warm fallback when a request comes
in against a cold backend.

Known limitation in the current code: the worker runs in-process. With
gunicorn's default 2 workers, each Flask process has its own worker
thread and its own queue, so two clients can effectively run jobs in
parallel. This is fine for low traffic, but if traffic ever grows, the
worker should move to a real job queue (Redis, etc.).

## Restarting Flask

After ingesting new content (stage 3) or changing `rag_controller.py`,
restart Flask so it re-boots with the new state:

```bash
# Local dev
pkill -f "gunicorn.*app:app"  # or just Ctrl-C if foregrounded
gunicorn -c chat_server/gunicorn.conf.py app:app

# docker-compose
docker compose restart web
```

Verify in the logs that the right DB path got loaded.

## Where to go next

[06-using.md](06-using.md) covers the user-facing side: subset combos,
modes, query patterns. The [codelab](codelab.md) walks through
spinning up Flask locally against a small DB.
