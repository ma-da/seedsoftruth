# Stage 6 — Using the System

This document is about *consuming* the running service — either via
the web UI at `seedsoftruth.peerservice.org` (production) or
`http://localhost:5000` (local dev), or programmatically against
the JSON API.

## The web UI

A single-page app in `templates/index.html` + `static/app.js`. The
visible controls are intentionally minimal; advanced controls are
gated behind dev mode.

### Modes

Three radio-style buttons at the top of the tools panel:

- **Search** — retrieval only. Fastest. No LLM call. The result list
  shows the top-K chunks with their snippets, BM25 scores, and a
  "View source" link. Use this when you want to see what the retriever
  is finding before deciding whether to ask the LLM.
- **AI Chat** — the full RAG pipeline. Retrieves, builds context,
  calls the LLM, returns a written answer with citations. The
  citations link back to the same chunks the Search mode would have
  shown. Slower (15–60 seconds depending on model and queue depth).
- **A/B test** — calls two LLM adapters in parallel with the same
  retrieved context. Shows both answers side-by-side. Used by the
  team to compare model variants. The two model types are configured
  in dev mode.

### Subset combos

The "Select Topic" dropdown contains named combos defined in
`static/app.js` `SUBSET_COMBOS`. Each combo expands to a list of
`subset_name` values that get sent to `/api/search` as the `subsets`
filter. Current combos:

| Combo | Subsets |
|---|---|
| WantToKnow | WTK Archive, WantToKnow.info, PEERS Substack |
| Deep Politics | WTK + theblackvault.com |
| Health | WTK + childrenshealthdefense.org, usrtk.org, vaccinepapers.org, howdovaccinescauseautism.org |
| UFO | WTK + theblackvault.com, newparadigminstitute.org |
| Everything | All ~19 subsets |

To add a new source to a combo, edit the `SUBSET_COMBOS` array in
`static/app.js` and refresh the page — no server restart needed for
that change.

### Algo & Prompt fields (dev mode only)

In `CFG.DEV_MODE = true`, the tools panel shows two extra fields:

- **Algo** — integer 1–5 selecting the retrieval algorithm. 5 is the
  production default (V5 = V4's min-gate + topic-match boost). Use 1
  or 2 to compare against a baseline; use 4 to test V5's boost
  contribution.
- **Prompt** — integer 1..N selecting the system prompt template from
  `model_prompts.MODEL_SYSTEM_PROMPTS`. Different prompts produce
  noticeably different tone and citation styles.

In production (`DEV_MODE: false`), both fields are hidden and the wire
defaults always send (`CFG.DEFAULT_RAG_ALGO_TYPE` = 5,
`CFG.DEFAULT_PROMPT_TYPE` = 1).

### Memory

The "Previous questions remembered" slider controls how many prior
Q/A turns get sent along with the current request, so the model can
maintain conversational context. Higher = longer effective prompt, so
more cost / latency. Default 3 turns.

### Reading the results panel

For Search mode (and the citations in AI Chat mode), each result card
shows:

```
Subset: WantToKnow.info
BM25: 21.21, entity_score: , fulltext_score: 21.21, raw_score: 21.21
[snippet of the matching chunk]
```

How to interpret:

- **Subset** — which source the chunk came from. If you only wanted
  PEERS Substack results, check that your combo actually includes it.
- **BM25** — the merged (hybrid) score the retriever assigned this
  chunk. Higher is better. For context, the production score floor is
  17.0 — anything below that on a non-phrase query would have been
  declined. Scores above ~25 are very strong matches; below ~17
  usually means the retrieval was finding *something* but it wasn't
  central.
- **entity_score** — present (non-blank) only when the entity-FTS
  branch produced a hit on this chunk. A blank means the chunk
  matched on body text only.
- **fulltext_score** — the per-row fulltext FTS contribution.
- **topic_boost / topic_score** (V5 results, not always rendered) —
  if `topic_boost` > 1.0, the chunk got a V5 topic-match bump; if
  exactly 1.0, no topic overlap. `topic_score = score_bm25 *
  topic_boost`.

The Sources link (the underlined "Subset: ..." text) opens the
original document in a new tab.

## Calling the API programmatically

The same endpoints used by the UI are open for direct use. Examples
below assume the service is reachable at `http://localhost:5000`.
Substitute the production base URL as needed.

### Retrieval-only search

```bash
curl -s -X POST http://localhost:5000/api/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "epstein financial connections",
    "top_k": 5,
    "shard_k": 20,
    "subsets": ["WTK Archive", "WantToKnow.info", "PEERS Substack"],
    "rag_algo_type": 5
  }' | python3 -m json.tool
```

`top_k` is the number of chunks to return. `shard_k` is an internal
parameter from the deprecated centroid-shard router; safe to pass 20.

### Full chat (async)

```bash
# 1) Enqueue the job
curl -s -X POST http://localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "abc123",
    "query": "What did the Senate panel find about geoengineering?",
    "history_turns": 3,
    "use_rag": true,
    "subsets": ["WTK Archive", "WantToKnow.info", "PEERS Substack"],
    "model_type": "hf",
    "rag_algo_type": 5,
    "prompt_type": 1
  }'
# → returns {"ok": true, "job_id": "...", "user_id": "abc123"}

# 2) Poll for the result
curl -s "http://localhost:5000/api/job/<job_id>"
# → eventually returns {"ok": true, "status": "done", "answer": "...",
#                       "references": [...]}
```

`model_type` can be `"hf"`, `"deepinfra"`, `"spark"`, or `"sim"` per
the adapter table in [05-serving.md](05-serving.md). `prompt_type` is
1-indexed.

### Health check

```bash
# Probe the server default model (set by MODEL_ADAPTER env var)
curl -s -X POST http://localhost:5000/api/status \
  -H 'Content-Type: application/json' \
  -d '{"health": "true"}'
# → {"ok": true, "model_ready": true, "unlocked": true, ...}

# Probe a specific adapter (matches what the UI does when the user
# picks a non-default model from the dropdown)
curl -s -X POST http://localhost:5000/api/status \
  -H 'Content-Type: application/json' \
  -d '{"health": "true", "model_type": "spark"}'
```

`model_type` is optional and accepts the same values as `/api/chat`.
This is what `tools/eval/prober.py` polls to surface degraded states.

## Query patterns and how to write better ones

The retriever has known strengths and weaknesses; understanding them
makes the difference between getting useful results and useless ones.

### Phrase matching is very strong

If you paste a verbatim sentence from a document, the phrase branch
will find that document immediately — and the V5 gate now lets phrase
matches through even when their BM25 looks low. So if you remember a
specific quote, paste it; you'll get the exact source.

### Specific named entities help a lot

Queries that include proper nouns ("Jeffrey Epstein", "Iran-Contra",
"Pat Buchanan", "Children's Health Defense") trigger the entity-FTS
branch, which is the most precise of the three. Including 1–3 specific
named entities usually beats a generic phrasing.

### Generic concept queries are weaker

Queries like `"What does the research say about consciousness?"` are
under-specified — no entities, vague verbs, no rare keywords. BM25
ranks them broadly and the topic boost can't always rescue it. Add
specifics: an author name, a study name, a publication.

### Off-corpus probes get rejected on purpose

The min-gate will decline queries that look like random word salad or
that aren't grounded in the corpus. If you genuinely don't know what
the corpus covers, run **Search mode** first to scope what's there
before asking a chat-style question. The gate decision is reported in
the response as `gate_decision` / `gate_reason`.

### Multiple subsets ≠ better

Picking the "Everything" combo on every query is tempting but not
always optimal. Narrower subset-scoping by combo gives the retriever a
cleaner, more topical pool to search, which can improve precision on
focused questions.

## Common UI/UX gotchas

- **Subset state persists in localStorage.** Switching combos updates
  it; switching off DEV_MODE doesn't reset the algo/prompt overrides.
  Clear browser storage if you see weird behavior.
- **"Found 1 result" with the wrong source** usually means the DB
  path is wrong (running Flask against a stale DB without the new
  ingest). The boot log line is your tell — see
  [05-serving.md](05-serving.md).
- **Chat takes forever, then errors out** — usually the selected
  model backend is cold-starting. Check `/api/status` with the same
  `model_type` you have selected in the UI (the dropdown sends it).
  If `model_ready=false`, the queue worker will warm that specific
  adapter and the job will run on the next cycle. If it's stuck for
  more than a minute or two, check the backend directly (HF: restart
  the endpoint from the HF UI; Spark: check the Cloudflare-fronted
  origin can answer `GET /health` within the 5-second timeout).

## Where to go next

If you want a hands-on tour of the pipeline producing results you
can interrogate in the UI, do the [codelab](codelab.md) next.
