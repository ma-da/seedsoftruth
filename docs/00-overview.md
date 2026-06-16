# Overview & ML Primer

This document explains what the system *is*, what each stage of the
pipeline produces and consumes, and the machine-learning vocabulary you
need before reading the rest of the docs.

## What Seeds of Truth does

A user types a question into the chat box. The system:

1. **Retrieves** a small number of relevant text passages from a curated
   SQLite database of ~62,000 pre-indexed passages drawn from
   ~19 source publications (WantToKnow.info, Children's
   Health Defense, the Black Vault, PEERS Substack, etc.).
2. **Builds a prompt** that contains the retrieved passages plus the
   user's question.
3. **Sends that prompt to a large language model** (LLM) which produces
   a textual answer grounded in the retrieved passages.
4. **Returns the answer + the source citations** so the user can verify
   each claim against the original document.

The architecture is called **retrieval-augmented generation (RAG)**.
Why we use it instead of just fine-tuning a model on the corpus, and
the trade-offs involved, are covered later in this doc.

## The five stages, end to end

| # | Stage | Input | Output | Updated |
|---|---|---|---|---|
| 1 | **Crawl** | A starting URL or sitemap | `.html` + `.txt` files on disk | When adding a source |
| 2 | **Clean** | Raw `.txt` from stage 1 | Cleaned `.txt` (boilerplate stripped) | When adding a source |
| 3 | **Ingest** | Cleaned `.txt` files | SQLite DB with chunks, entities, topics, FTS5 indexes | When adding a source |
| 4 | **Train** | Curated Q&A pairs derived from the corpus | LoRA adapter weights + HF endpoint | Rarely (months) |
| 5 | **Serve** | A user's question | A grounded answer + citations | Always running |

Stages 1–3 and 5 are independent — you can run them whenever the corpus
changes. Stage 4 (fine-tuning) is heavy, requires a GPU, and is run
only when the team decides to refresh the model. **Most updates to the
system involve only stages 1–3 plus a Flask restart in stage 5.**

## ML primer — the vocabulary

If terms like "embedding", "BM25", "LoRA", or "fine-tune" are unfamiliar,
read this section before continuing. Each concept is explained in the
context of where it appears in *this* codebase.

### What is a Large Language Model (LLM)?

An LLM is a neural network — roughly, a giant mathematical function
parameterized by billions of numbers (called *weights*). You give it a
sequence of text tokens (~words) called the **prompt**, and it produces
a probability distribution over what the next token should be. Sample
from that distribution, append the token, and repeat — you've generated
text.

The base LLM used in this project is **Qwen2.5-7B-Instruct** ("7B" =
7 billion parameters). It's been trained by Alibaba's research team on
a broad mix of internet text, then instruction-tuned to follow
human-written prompts. The Seeds of Truth team then **fine-tunes a LoRA
adapter** on top of it (see below) to teach it to engage with
specific source material in a specific tone.

### Why fine-tuning alone isn't enough — and why we need RAG

You might wonder: if we fine-tune the model on our 166k-passage corpus,
won't it just *know* the contents? Why retrieve passages at all?

Two reasons:

1. **Models hallucinate.** Even after training on a corpus, a model
   often produces statements that *sound* like things in the corpus but
   aren't actually attested anywhere. For a project whose entire value
   proposition is *citable, verifiable sources*, that's catastrophic.
   By forcing the model to ground its answer in passages we hand it at
   query time, we can show the user exactly which passage each claim
   came from.
2. **The corpus changes.** Every time the team adds a new Substack post
   or a new source document, fine-tuning would have to be redone (hours
   of GPU compute + dataset prep). Adding a passage to the retrieval
   DB takes minutes and requires no GPU.

So we do **both**: light fine-tuning teaches the model house tone,
domain vocabulary, and how to handle contested claims responsibly
(see [04-training.md](04-training.md)); RAG handles the factual content
at query time.

### What is a chunk?

LLM context windows have a maximum size — Qwen2.5-7B handles a few
thousand tokens at most. We can't shove an entire 30-page article in
front of the model and expect it to read it. So during **ingest** (stage
3), every source document is broken into overlapping segments called
**chunks** of roughly 250–400 words each.

The retrieval system finds the *most relevant chunks* — not the most
relevant documents. A 30-page book might contribute one
chunk to a given query and nothing for another. This is how a
SQLite DB with ~62k chunks fits inside a 4k-token prompt context.

Each chunk in the DB has:
- A unique integer `lookup_id` (the SQLite primary key)
- A string `chunk_id` (a content hash — used for deduplication)
- A `subset_name` (which publication it came from — `"WTK Archive"`,
  `"PEERS Substack"`, etc.)
- The text itself (`fulltext_text`)
- Pre-extracted named entities (`entities_text`)
- Optional topic tags (joined via `chunk_topics`)
- A `source_url` so the UI can link back to the original article

The schema and chunking strategy are detailed in [03-ingest.md](03-ingest.md).

### How retrieval works: BM25, not embeddings

There are two main families of text retrieval today:

- **Lexical / sparse retrieval (BM25)**: scores documents by how often
  the query's *literal words* appear in them, weighted by how rare each
  word is across the corpus (rare words are stronger signals).
- **Dense / embedding retrieval**: trains a neural network to map text
  to a fixed-size vector, then finds documents whose vector is closest
  to the query's vector. This can catch synonyms and paraphrases that
  BM25 misses (`"covid origins"` ≈ `"SARS-CoV-2 emergence"`).

Seeds of Truth currently uses **BM25 only**, via SQLite's built-in FTS5
full-text-search extension. Reasons:

- **No vector DB infrastructure to maintain.** Everything lives in a
  single SQLite file.
- **Determinism.** BM25 ranking is exactly reproducible. Embedding
  models drift between versions; FTS5 doesn't.
- **Corpus topics use named entities heavily** ("Epstein", "Iran-Contra",
  "geoengineering"). These are proper nouns that BM25 handles well —
  embedding models are more useful when the user paraphrases concepts.
- **A second FTS5 index over a pre-extracted `entities_text` column**
  gives us most of the precision benefit of NER-aware retrieval without
  the complexity of a vector store. This is what the code calls
  *hybrid retrieval* — see [05-serving.md](05-serving.md).

Whether to add dense retrieval later is an open question. The retrieval
quality eval in `eval/ab_eval.py` is set up to make that comparison
quantitative when the time comes.

### What is named entity recognition (NER)?

NER is a small NLP model that scans a piece of text and labels spans
that look like proper nouns: people, organizations, places, dates,
events, works. We use **spaCy's `en_core_web_sm`** model — small,
fast, runs on CPU.

NER is used in two places:

1. **At ingest time** (`corpus_to_hybrid_db.py`): NER tags each chunk's
   text, and the resulting entity strings get indexed into
   `entities_fts`. A query for `"Jeffrey Epstein"` then hits this
   entity-aware index in parallel with the regular fulltext index.
2. **At query time** (`rag_controller.extract_canonical_entity_terms_typed`):
   the same NER scans the user's query to pull out entity terms, which
   become explicit search terms against `entities_fts`. This is also
   how the min-gate decides whether a query has "real entities" (see
   the V4/V5 min-gate in [05-serving.md](05-serving.md)).

### What is a topic?

A *topic* in this codebase is a coarse-grained category attached to a
chunk — things like `health_science::vaccines_and_immunization` or
`media_information::censorship_and_content_moderation`. They're
produced (optionally) at ingest time by a separate BGE + LLM classifier
that the team runs on GPUs. There are ~172 topics across 16 domains.

Topics don't affect retrieval at the FTS5 layer. They show up in
**V5**, the production retrieval algorithm, which gives chunks a
multiplicative score boost when their topics overlap with the query's
inferred topics. See [05-serving.md](05-serving.md) for the V5 details.

### What is fine-tuning? What is LoRA?

**Fine-tuning** is the process of taking a pre-trained LLM and nudging
its weights so it behaves differently on a specific kind of input.
The pre-trained Qwen2.5-7B already knows English; fine-tuning teaches
it the *house style* — how to engage with deep-politics topics
responsibly, how to format citations, what NOT to say in cases of low
evidence.

Fine-tuning every weight in a 7-billion-parameter model is expensive
(many GPUs, hours of compute, hundreds of GB of disk for one snapshot).

**LoRA** (Low-Rank Adaptation) is a trick that fine-tunes only a tiny
fraction of the weights — typically less than 1%. Instead of modifying
the base model's matrices, you train small "adapter" matrices that get
*added* to the base at inference time. A LoRA adapter for Qwen2.5-7B
is ~50–200 MB; training one is hours-not-days on a single GPU.

This codebase produces LoRA adapters, not full fine-tunes. See
[04-training.md](04-training.md) for the actual recipe.

### What does the served model actually look like?

When you hit the chat endpoint in production, here's what happens
under the hood:

1. The Flask app loads a **retrieval state** (the SQLite DB + the
   spaCy NER model + an entity-canonicalization map).
2. A user question comes in.
3. `search_references()` retrieves the top-K chunks (default K=20).
4. `build_context()` assembles those chunks into a prompt block.
5. The prompt block + a system prompt + the user's question goes to
   one of four **LLM adapters**:
   - `HFEndpointLLM` — calls a Hugging Face Inference Endpoint
     running Qwen + the project's LoRA adapter. **This is production.**
   - `DeepInfraLlamaLLM` — calls DeepInfra's hosted Llama-3.
   - `SparkCloudflareLLM` — calls a self-hosted model behind
     Cloudflare Access.
   - `SimEndpointLLM` — a fake model that just echoes the prompt back,
     used for tests.
6. The model streams back tokens; the app cleans them up, attaches the
   citations from step 3, and returns JSON to the browser.

Step 1 and 5 happen once per Flask process; steps 2–4 happen per request.

## Repo layout

Top-level directories in the seedsoftruth checkout:

```
seedsoftruth/
├── chat_server/            The Flask web server (run from repo root with
│   │                       chat_server/ on PYTHONPATH — see scripts/run.sh)
│   ├── app.py              Flask app — HTTP endpoints
│   ├── rag_controller.py   Retrieval + context-build + ask()
│   ├── model_adapters.py   LLM adapter classes (HF / DeepInfra / Spark / Sim)
│   ├── model_prompts.py    System prompts (DEEP_REPORTING_V1/V2/V3, SMOKING_MAN)
│   ├── db.py               SQLite (jobs + feedback — separate from corpus DB)
│   ├── gunicorn.conf.py    gunicorn config (adds chat_server/ to import path)
│   ├── static/             Frontend (vanilla JS + CSS)
│   │   └── app.js          Subset combos, query dispatch, result rendering
│   └── templates/
│       └── index.html      The single-page app shell
│
├── webscraper/             Stage 1 — see docs/01-crawl.md
│   ├── web_scraper.py
│   ├── substack_scrape.sh
│   └── corpus*/            Output: HTML/TXT files
│
├── tools/                  Stages 2 + 3 — see docs/02-cleaning.md, 03-ingest.md
│   ├── clean_web_corpus.py
│   ├── text_cleaners.py
│   ├── corpus_to_hybrid_db.py
│   └── clean_chunks.py
│
├── train_ai/               Stage 4 — see docs/04-training.md
│   ├── gen_qa_pairs_*.py   Dataset prep
│   ├── handler.py          HF Inference Endpoint handler
│   └── inference_test_*.ipynb  Eval notebooks
│
├── data/
│   ├── gamma_master_hybrid_fts_stage3.db   The corpus DB (1.8 GB)
│   └── entity_query_normalization_map.flat.json   Entity alias map
│
├── eval/                   Retrieval evaluation harness
│   ├── ab_eval.py
│   └── labels.jsonl
│
├── tests/                  pytest unit + integration tests
│
└── docs/                   You are here.
```

The Flask process starts via `gunicorn` (config in `chat_server/gunicorn.conf.py`),
typically with 2 workers. A docker-compose.yml wires up the production
deployment.

## Where to go next

If this is your first time, do the [codelab](codelab.md) now — running
the pipeline end-to-end on a tiny dataset makes everything below
concrete. The stage docs ([01-crawl.md](01-crawl.md) onward) are
reference material to read alongside the codelab or before contributing
changes to a specific stage.
