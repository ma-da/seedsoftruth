# Seeds of Truth — Documentation

Seeds of Truth is a retrieval-augmented question-answering system over a
curated corpus of investigative journalism and primary-source material on
contested topics (deep politics, public health controversies, UFOs/UAPs,
declassified intelligence, etc.). It is built and run by PEERS / the
WantToKnow.info team. This documentation explains every stage of the
pipeline, from pulling raw web pages off the public internet through to
answering a user's question in the web UI.

## Who this is for

These docs are written for two audiences at once:

- **Someone joining the project** who can read Python but has never built
  a RAG system and doesn't know terms like *chunk*, *BM25*, or *LoRA*.
- **Someone already comfortable with ML** who wants a concrete map of
  *this* codebase — which file does what, which knobs matter, what the
  defaults are tuned for.

We assume basic familiarity with the command line, Python, and Git. We
do **not** assume familiarity with machine learning, NLP, information
retrieval, or full-stack web development. Background concepts are
explained inline the first time they appear, and concentrated in
[00-overview.md](00-overview.md).

## Pipeline at a glance

```
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ 1. Crawl │ → │ 2. Clean │ → │ 3.Ingest │ → │ 4. Train │ → │ 5. Serve │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
   web pages      stripped       SQLite DB +    fine-tuned       Flask +
   as HTML/TXT    article TXT    FTS5 index     LoRA adapter    LLM adapters
                                                                     │
                                                                     ▼
                                                              ┌────────────┐
                                                              │ 6. Use it  │
                                                              │ (UI / API) │
                                                              └────────────┘
```

Each stage has its own document below. Stages 1–3 and 5–6 are run every
time the corpus changes. Stage 4 (fine-tuning the language model itself)
is run rarely — the served LLM is typically a base model plus a static
LoRA adapter built on a snapshot of the corpus.

## File index

| File | What's in it |
|---|---|
| [00-overview.md](00-overview.md) | System architecture + ML primer (what a chunk is, why RAG exists, etc.). **Start here if you've never built a RAG.** |
| [01-crawl.md](01-crawl.md) | `webscraper/web_scraper.py`, Playwright vs requests, `--scroll` for infinite-scroll sites, `substack_scrape.sh` wrapper. |
| [02-cleaning.md](02-cleaning.md) | `tools/clean/clean_web_corpus.py`, file-level rejection (JS stubs, listing pages), block-level cleaning, repeated-footer cuts. |
| [03-ingest.md](03-ingest.md) | `tools/index/corpus_to_hybrid_db.py` — chunking, the SQLite schema, NER, topic classification, BM25 FTS5 indexing. |
| [04-training.md](04-training.md) | LoRA fine-tuning concepts, the `train_ai/` scripts, dataset format, when fine-tuning is and isn't the right answer. |
| [05-serving.md](05-serving.md) | The Flask app + gunicorn, retrieval algos 1–5 (V5 is the production default), model adapters (HF / DeepInfra / Spark / Sim), `/api/search` and `/api/chat` endpoints. |
| [06-using.md](06-using.md) | The web UI: subset combos, search vs chat modes, query patterns. API examples for programmatic use. Interpreting result scores. |
| [codelab.md](codelab.md) | **Hands-on walkthrough.** Crawl 5 Substack posts, clean them, ingest into a fresh local DB, query directly via `rag_controller`, then optionally via the Flask API. ~30 minutes end-to-end, no GPU required. |

## Reading order

- **First time:** [00-overview.md](00-overview.md) → [codelab.md](codelab.md) → revisit individual stage docs as needed.
- **Adding a new corpus source:** [01-crawl.md](01-crawl.md) → [02-cleaning.md](02-cleaning.md) → [03-ingest.md](03-ingest.md).
- **Debugging retrieval:** [03-ingest.md](03-ingest.md) (schema) → [05-serving.md](05-serving.md) (retrieval algos) → [06-using.md](06-using.md) (query patterns).
- **Deploying a different LLM:** [04-training.md](04-training.md) → [05-serving.md](05-serving.md) (the adapters section).
