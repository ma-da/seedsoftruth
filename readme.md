# Seeds of Truth

**A grounded-answer RAG system for question-answering over your own curated corpus, with multiple LLM backends and a built-in evaluation harness.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

> **Status:** Alpha. Reliable enough for invited testers; not yet hardened for the public internet. See [Project status](#project-status) and [SECURITY.md](SECURITY.md).

---

## What it is

Seeds of Truth is a self-hosted retrieval-augmented question-answering system. Drop in a corpus of source documents, configure an LLM backend, and get answers with citations back to the source material.

The system is designed to surface evidence that lives in long-form, lightly-indexed corpora — investigative archives, primary documents, government reports, transcripts — where conventional web search underperforms. The reference deployment runs over a corpus of long-form journalism and primary sources on contested historical and policy topics, but the codebase is corpus-agnostic: anything that ingests into the SQLite FTS5 schema works.

Live reference deployment: [seedsoftruth.peerservice.org](https://seedsoftruth.peerservice.org)

## Highlights

- **Hybrid SQLite FTS5 retrieval** — combines BM25 over entity-tagged metadata (`entity_fts`) and full-text content (`fulltext_fts`), with weighted score fusion and a calibrated minimum-relevance gate that declines to answer when the corpus has no good evidence.
- **Multi-backend LLM strategy** — pluggable adapter layer (`LLMStrategy` ABC) shipping with four backends: HuggingFace Inference Endpoints, DeepInfra (Llama-3), a Cloudflare-Access-fronted internal endpoint ("Spark"), and a deterministic test simulator.
- **Configurable system prompts** — multiple prompt variants ship in `model_prompts.py`; runtime selection per request. Default prompt classifies questions as established / contested / anomalous / low-evidence and shapes the response accordingly.
- **Built-in evaluation harness** — labeler, A/B comparison, judge-cache, and metric reporting under `eval/`. Wire in your own seed queries; rerun after every retrieval-algorithm change.
- **Single-binary deploy** — Flask + gunicorn + SQLite. No external services beyond the LLM backend and (optionally) HuggingFace for entity-aware NER.
- **Job queue with SQLite-backed persistence** — long-running LLM calls are queued and polled, so the UI doesn't block.

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │            Browser (SPA)             │
                    │       static/app.js · index.html     │
                    └──────────────────┬───────────────────┘
                                       │ HTTPS / JSON
                                       ▼
                    ┌──────────────────────────────────────┐
                    │      Flask + gunicorn (app.py)       │
                    │   /api/search · /api/chat · /api/*   │
                    └────┬───────────────────┬─────────────┘
                         │                   │
                         ▼                   ▼
                    ┌─────────────┐   ┌──────────────────┐
                    │   db.py     │   │ rag_controller.py│
                    │  app jobs   │   │  retrieval +     │
                    │  + feedback │   │  context build   │
                    └─────┬───────┘   └────────┬─────────┘
                          │                    │
                          ▼                    ▼
                    ┌─────────────┐   ┌──────────────────┐
                    │  app.db     │   │  hybrid_fts.db   │
                    │  (SQLite)   │   │  (SQLite FTS5)   │
                    └─────────────┘   └────────┬─────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │   model_adapters.py  │
                                    │   LLMStrategy ABC    │
                                    └──┬─────────┬─────────┘
                                       │         │
                                       ▼         ▼
                               HuggingFace   DeepInfra
                                Endpoint      (Llama-3)
                                       │         │
                                       └────┬────┘
                                            ▼
                                       LLM response
```

A request to `/api/chat` flows: input validation → rate limiter → DB job insert → readiness check → either inline `chat_with_corpus()` or queue for the worker thread → hybrid retrieval (`rag_controller`) → context assembly → LLM call (one or two, depending on `USE_DOUBLE_PROMPT`) → reference cleaning → response.

## Quick start

> The fastest path is `docker compose up`, but containerization is on the [roadmap](#roadmap), not landed yet. The instructions below run the app directly.

### Prerequisites

- Python 3.11 (see `.python-version`)
- `gcc` and `make` for the spaCy model wheel
- An LLM backend of your choice (HuggingFace Inference Endpoint, DeepInfra account, etc.)
- A corpus database in the expected SQLite FTS5 schema (see [Bring your own corpus](#bring-your-own-corpus))

### Install

```bash
git clone https://github.com/<your-org>/seedsoftruth.git
cd seedsoftruth

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Minimum configuration

Create a `.env` file (or export these in your shell):

```bash
# Required — Flask session signing
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"

# Required — comma-separated allow-list for the chat password gate
export SOT_PASSWORDS="my-shared-password"

# Required — pick at least one LLM backend
export MODEL_ADAPTER="deepinfra"        # or "hf" | "spark" | "sim"
export DEEPINFRA_TOKEN="..."            # if using deepinfra (web app)
export HF_API_KEY="..."                 # if using huggingface

# Required — path to the corpus database
export HYBRID_DB_PATH="./data/your_corpus.db"
```

A more complete reference is in `CONFIG.md` *(planned — see roadmap)*.

### Run

Development:

```bash
flask --app app run
```

Production:

```bash
gunicorn -c gunicorn.conf.py app:app
```

Open http://localhost:5000 — the search endpoint (`/api/search`) is open; chat (`/api/chat`) prompts for the password from `SOT_PASSWORDS`.

### Run the eval harness

```bash
# One-time bootstrap (judges queries with the LLM judge — costs API credits)
./scripts/bootstrap_eval.sh

# A/B comparison between retrieval algorithm variants
./scripts/run_eval.sh
```

## Configuration reference (key environment variables)

| Variable                  | Default                                | Purpose                                              |
|---------------------------|----------------------------------------|------------------------------------------------------|
| `FLASK_SECRET_KEY`        | *(required, no default)*               | Flask session signing key. Set to a long random hex. |
| `SOT_PASSWORDS`           | `""`                                   | Comma-separated allow-list for the chat password gate. |
| `MODEL_ADAPTER`           | `hf`                                   | One of: `hf`, `deepinfra`, `spark`, `sim`.           |
| `HF_API_KEY`              | `""`                                   | HuggingFace Inference Endpoint API key.              |
| `HF_TIMEOUT_SECS`         | `900`                                  | Per-LLM-call timeout. **Lower this in production.**  |
| `HF_MAX_ATTEMPTS`         | `10`                                   | LLM retry count on 503. **Lower this in production.**|
| `DEEPINFRA_TOKEN`         | `""`                                   | DeepInfra API token (web app). The eval harness reads it from `DEEPINFRA_API_KEY` instead — see [CONFIG.md](CONFIG.md). |
| `HYBRID_DB_PATH`          | `./data/gamma_master_hybrid_fts_stage2.db` | Path to the corpus FTS5 SQLite database.         |
| `ENTITY_CANON_MAP_PATH`   | `./data/entity_query_normalization_map.flat.json` | Entity canonicalization map.            |
| `SPACY_MODEL`             | `en_core_web_sm`                       | spaCy model for query NER.                           |
| `HYBRID_ENTITY_WEIGHT`    | `4.0`                                  | Weight on the entity-FTS branch in score fusion.     |
| `HYBRID_FULLTEXT_WEIGHT`  | `1.0`                                  | Weight on the fulltext-FTS branch in score fusion.   |
| `MIN_GATE_SCORE_FLOOR`    | `17.0`                                 | BM25 floor below which the system declines to answer (when min-gating is enabled). |

## Bring your own corpus

The retrieval layer expects a SQLite database with FTS5 virtual tables matching the schema in `data/`. At minimum the schema includes:

- `chunks` — passage-level rows with `chunk_id`, `lookup_id`, source metadata, and full text
- `entity_fts` — FTS5 virtual table indexed over canonicalized entity names per chunk
- `fulltext_fts` — FTS5 virtual table indexed over the chunk text
- `entities`, `chunk_entities`, `topics`, `chunk_topics` — relational tables for entity and topic metadata

A reference ingestion pipeline is *not yet open-sourced* (see [roadmap](#roadmap)). For now, `db/metadata.sql` and `db/load_metadata.py` document the schema; treat them as authoritative.

## Project structure

```
.
├── app.py                  # Flask routes, auth gate, worker thread bootstrap
├── rag_controller.py       # Retrieval pipeline, context build, ask()
├── model_adapters.py       # LLMStrategy ABC + four concrete backends
├── model_prompts.py        # System prompt variants
├── rag_cleaner.py          # Text and entity cleanup helpers
├── db.py                   # SQLite jobs + feedback (separate from corpus DB)
├── utils.py                # Cross-cutting helpers (rate limiting, payload parsing)
├── logging_config.py       # Logger setup
├── gunicorn.conf.py        # Gunicorn config (production entry point)
├── static/                 # Frontend SPA — app.js, style.css, fonts, images
├── templates/              # Jinja templates (single-page app shell)
├── data/                   # Corpus databases and entity-canon maps
├── db/                     # App database (jobs, feedback) and schema files
├── eval/                   # Evaluation harness — labeler, A/B comparison
└── scripts/                # Deploy, eval, and ad-hoc test scripts
```

## Methodology and corpus

The reference corpus and default system prompts make editorial choices that are worth being explicit about. The codebase is corpus-agnostic — none of the retrieval or LLM-orchestration code is opinionated about subject matter — but the prompts shipped in `model_prompts.py` shape responses on contested topics by:

- Asking the model to classify a question as Established, Contested, Anomalous, or Low-evidence, rather than always defending a mainstream consensus.
- Surfacing competing claims when they appear in the retrieved evidence.
- Ending responses on contested topics with a "Plausibility Spectrum" (Strongly Supported / Moderately Supported / Indeterminate / Weakly Supported / Speculative / Disputed).

These prompt designs reflect a research-tool stance: the system is intended to surface evidence rather than enforce a single narrative. They are also configurable — operators can swap in their own prompts in `model_prompts.py` and select between them per request via `prompt_type` on `/api/chat`. See `model_prompts.py` for the full set of shipped variants.

## Project status

**This is alpha software.** It runs in production for an invited group of testers, but it is not yet hardened for the open internet. Known gaps tracked in `ALPHA_BLOCKERS.md` and being worked through:

- No CSRF protection on POST endpoints
- No per-IP rate limit on `/api/unlock`
- LLM call timeouts not yet bounded by a wall-clock cap
- Job queue persists in SQLite for inserts but is read in-process — losing in-flight jobs across worker restarts
- Test coverage is shell-script integration tests; no pytest suite yet

Before deploying publicly, read [SECURITY.md](SECURITY.md) *(planned)* and address the items in `ALPHA_BLOCKERS.md`.

## Roadmap

Near-term (path to v1.0):

- `docker compose up` quickstart with a small public sample corpus
- pytest suite + GitHub Actions CI (ruff, mypy, pytest, pip-audit)
- Centralized configuration module (pydantic-settings) and a `CONFIG.md` reference
- Worker process split out of the Flask process; SQLite-backed queue as source of truth
- Open-source the ingestion pipeline so users can build a corpus without reverse-engineering the schema
- Hardened security defaults (CSRF, rate-limit on auth endpoint, scrubbed logs)

Longer-term:

- Embeddings layer (currently BM25-only) for semantic recall on top of the lexical hybrid
- Multi-tenant deployment mode with per-user accounts (current model is a shared password gate)
- Web UI for corpus management and prompt editing

## Contributing

We welcome bug reports, feature suggestions, and pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) *(planned)* for development setup, test commands, and PR conventions.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

Built on top of [spaCy](https://spacy.io/), [bm25s](https://github.com/xhluca/bm25s), [Flask](https://flask.palletsprojects.com/), [SQLite FTS5](https://www.sqlite.org/fts5.html), and the model providers integrated through `model_adapters.py`. The reference deployment uses corpora from independent journalism and primary-source archives, made available by their respective publishers.
