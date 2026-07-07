# Seeds of Truth

**A grounded-answer RAG system for question-answering over your own curated corpus, with multiple LLM backends and a built-in evaluation harness.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

> **Status:** Alpha. Reliable enough for invited testers; not yet hardened for the public internet. See [Project status](#project-status) and [SECURITY.md](SECURITY.md).

---

## What it is

Seeds of Truth is a self-hosted retrieval-augmented question-answering system. Drop in a corpus of source documents, configure an LLM backend, and get answers with citations back to the source material.

The system is designed to surface evidence that lives in long-form, lightly-indexed corpora — investigative archives, primary documents, government reports, transcripts — where conventional web search underperforms. The reference deployment runs over a corpus of long-form journalism and primary sources on contested historical and policy topics, but the codebase is corpus-agnostic: anything that ingests into the SQLite FTS5 schema works.

Live reference deployment: [seedsoftruth.peerservice.org](https://seedsoftruth.peerservice.org)

## Documentation

Full end-to-end documentation now lives under [`docs/`](docs/README.md), covering every pipeline stage from crawling raw pages to serving answers. It is written for two audiences at once — people new to RAG (terms like *chunk*, *BM25*, and *LoRA* are explained inline) and people who already know ML but want a concrete map of this codebase.

- [`docs/00-overview.md`](docs/00-overview.md) — system architecture and an ML primer. **Start here if you've never built a RAG system.**
- [`docs/01-crawl.md`](docs/01-crawl.md) · [`02-cleaning.md`](docs/02-cleaning.md) · [`03-ingest.md`](docs/03-ingest.md) — building a corpus: crawl → clean → SQLite FTS5 index.
- [`docs/04-training.md`](docs/04-training.md) — optional LoRA fine-tuning (`train_ai/`).
- [`docs/05-serving.md`](docs/05-serving.md) — the Flask app, retrieval algorithms 1–5 (V5 is the production default), and the model adapters.
- [`docs/06-using.md`](docs/06-using.md) — the web UI, search vs chat modes, query patterns, and API examples.
- [`docs/codelab.md`](docs/codelab.md) — **hands-on walkthrough:** crawl five posts, clean and ingest them into a fresh local DB, then query it. ~30 minutes, no GPU required. The fastest way to see the whole stack work.
- [`docs/CONFIG.md`](docs/CONFIG.md) — authoritative environment-variable reference (the table below is a quickstart subset).

## Highlights

- **Hybrid SQLite FTS5 retrieval** — combines BM25 over entity-tagged metadata (`entity_fts`) and full-text content (`fulltext_fts`), with weighted score fusion and a calibrated minimum-relevance gate that declines to answer when the corpus has no good evidence.
- **Multi-backend LLM strategy** — pluggable adapter layer (`LLMStrategy` ABC) shipping with five backends: HuggingFace Inference Endpoints, DeepInfra (Llama-3), a Cloudflare-Access-fronted internal endpoint ("Spark"), a streaming vLLM/OpenAI-compatible adapter (`vllm`), and a deterministic test simulator (`sim`). The backend is chosen per request via the `model_type` field on `/api/chat`.
- **Configurable system prompts** — multiple prompt variants ship in `model_prompts.py`; runtime selection per request. Default prompt classifies questions as established / contested / anomalous / low-evidence and shapes the response accordingly.
- **Built-in evaluation harness** — labeler, A/B comparison, judge-cache, and metric reporting under `eval/`. Wire in your own seed queries; rerun after every retrieval-algorithm change.
- **Single-binary deploy** — Flask + gunicorn + SQLite. No external services beyond the LLM backend and (optionally) HuggingFace for entity-aware NER.
- **Job queue with SQLite-backed persistence** — long-running LLM calls are queued and polled (`GET /api/job/<id>`), so the UI doesn't block.
- **Email-when-ready** — when a request is queued because the model is cold-booting, users can opt to be emailed the answer instead of waiting in-tab (`/api/email_response`). Gated by `SOT_EMAIL_RESPONSES_ENABLED` plus SMTP config; off until both are set.

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
                    │   routes.py blueprint · runtime.py   │
                    │   /api/search · /api/chat · /api/*   │
                    └────┬───────────────────┬─────────────┘
                         │                   │
                         ▼                   ▼
                    ┌─────────────┐   ┌──────────────────┐
                    │   db.py     │   │   corpus.py +    │
                    │  app jobs   │   │ rag_retrieval.py │
                    │  + feedback │   │  + rag_context   │
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
                                    └──┬────────┬────────┬──┘
                                       │        │        │
                                       ▼        ▼        ▼
                               HuggingFace  DeepInfra  vLLM / Spark
                                Endpoint     (Llama-3)  (streaming)
                                       │        │        │
                                       └────────┼────────┘
                                                ▼
                                          LLM response
```

A request to `/api/chat` flows: input validation → rate limiter (`runtime.py`) → DB job insert → readiness check → either inline `corpus.chat_with_corpus()` or queue for the background worker (`worker.py`) → hybrid retrieval (`rag_retrieval.py`) → context assembly (`rag_context.py`) → LLM call (one or two, depending on `USE_DOUBLE_PROMPT`) → reference cleaning → response. The route handlers live in `routes.py`; configuration is centralized in `config.py`; the unlock/auth gate is in `auth.py`; the shared retrieval engine state lives in `state.py`.

## Quick start

Two supported paths: **Docker Compose** (recommended for fresh deployments) and **direct Python** (for development against a checkout).

### Prerequisites (both paths)

- An LLM backend of your choice (HuggingFace Inference Endpoint, DeepInfra account, etc.)
- A corpus database in the expected SQLite FTS5 schema (see [Bring your own corpus](#bring-your-own-corpus))

### Path A — Docker Compose (recommended)

Prerequisites: Docker 24+ and Docker Compose v2.

```bash
git clone https://github.com/<your-org>/seedsoftruth.git
cd seedsoftruth

# Copy the env template and fill it in (at minimum: FLASK_SECRET_KEY,
# SOT_PASSWORDS, and credentials for whichever LLM backend you'll use).
cp deploy/.env.example deploy/.env
$EDITOR deploy/.env

# Place your corpus DB at data/gamma_master_hybrid_fts_stage3.db
# (or update HYBRID_DB_PATH in docker-compose.yml).

docker-compose up --build
```

App is then served at `http://localhost:8000`. See [`deploy/README.md`](deploy/README.md) for day-to-day commands and troubleshooting.

### Path B — Direct Python

Prerequisites:

- Python 3.11 (see `.python-version`)
- `gcc` and `make` for the spaCy model wheel

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
export MODEL_ADAPTER="deepinfra"        # or "hf" | "spark" | "vllm" | "sim"
export DEEPINFRA_TOKEN="..."            # if using deepinfra (web app)
export HF_API_KEY="..."                 # if using huggingface
# Tip: MODEL_ADAPTER="sim" needs no credentials — a deterministic stub for
# local dev so you can exercise the full stack without an LLM backend.

# Required — path to the corpus database
export HYBRID_DB_PATH="./data/your_corpus.db"
```

The full, authoritative environment-variable reference is in [`docs/CONFIG.md`](docs/CONFIG.md). The table below is a quickstart subset.

### Run

Development (run from the repo root; the web modules live in `chat_server/`
and import each other flat, so put that dir on the path):

```bash
PYTHONPATH=chat_server flask --app app run
# or simply: bash scripts/run.sh
```

Production:

```bash
gunicorn -c chat_server/gunicorn.conf.py app:app
```

Open http://localhost:5000 — the search endpoint (`/api/search`) is open; chat (`/api/chat`) prompts for the password from `SOT_PASSWORDS`.

### Run the eval harness

```bash
# One-time bootstrap (judges queries with the LLM judge — costs API credits)
./scripts/bootstrap_eval.sh

# A/B comparison between retrieval algorithm variants
./scripts/run_eval.sh
```

### Run the tests

A `pytest` suite lives under `tests/` (async chat DB/routes/worker, the retrieval min-gate, and the vLLM adapter):

```bash
PYTHONPATH=chat_server pytest
```

## Configuration reference (key environment variables)

| Variable                  | Default                                | Purpose                                              |
|---------------------------|----------------------------------------|------------------------------------------------------|
| `FLASK_SECRET_KEY`        | *(required, no default)*               | Flask session signing key. Set to a long random hex. App refuses to start without it (except under `FLASK_DEBUG`). |
| `SOT_PASSWORDS`           | `""`                                   | Comma-separated allow-list for the chat password gate. |
| `SOT_MAX_UNLOCK_ATTEMPTS` | `3`                                    | Wrong-password attempts per session before lockout (clamped to ≥1). |
| `SOT_IP_LOCKOUT_SECS`     | `86400`                                | How long an IP stays locked out after too many failed unlocks. Capped at 60s under `FLASK_DEBUG`. |
| `SOT_TRUSTED_PROXIES`     | `""`                                   | Comma-separated IPs/CIDRs whose `X-Forwarded-For` is trusted for client-IP. Empty = trust no headers. Set this in production behind a proxy. Replaces the deprecated `SOT_TRUST_PROXY_HEADERS`. |
| `SOT_ADMIN_TOKEN`         | `""`                                   | Token for remote `POST /api/admin/clear-lockouts`. Unset = local-only (loopback). |
| `MODEL_ADAPTER`           | `hf`                                   | One of: `hf`, `deepinfra`, `spark`, `vllm`, `sim`. Overridable per request via `model_type` on `/api/chat`. |
| `HF_API_KEY`              | `""`                                   | HuggingFace Inference Endpoint API key.              |
| `HF_TIMEOUT_SECS`         | `900`                                  | Per-LLM-call timeout. **Lower this in production (≈120).**  |
| `HF_MAX_ATTEMPTS`         | `10`                                   | LLM retry count on 503. **Lower this in production (≈3).**|
| `HF_MAX_WAIT_SECS`        | `6`                                    | Max backoff between retries on a HuggingFace `estimated_time` hint. |
| `DEEPINFRA_TOKEN`         | `""`                                   | DeepInfra API token (web app). The eval harness reads it from `DEEPINFRA_API_KEY` instead — see [docs/CONFIG.md](docs/CONFIG.md). |
| `SOT_VLLM_BASE_URL` etc.  | *(falls back to `SPARK_*`)*            | vLLM/OpenAI-compatible streaming backend. `SOT_VLLM_*` vars fall back to their `SPARK_*` equivalents when unset — see [docs/CONFIG.md](docs/CONFIG.md). |
| `SOT_EMAIL_RESPONSES_ENABLED` | `1`                                | Master flag for email-when-ready; the feature also requires SMTP config (`SOT_SMTP_HOST`, `SOT_EMAIL_FROM_ADDRESS`, ...) before it activates. |
| `RETRIEVAL_BACKEND`       | `sqlite_hybrid`                        | Retrieval engine selector. Only `sqlite_hybrid` is currently supported. |
| `HYBRID_DB_PATH`          | `./data/gamma_master_hybrid_fts_stage3.db` | Path to the corpus FTS5 SQLite database.         |
| `ENTITY_CANON_MAP_PATH`   | `./data/entity_query_normalization_map.flat.json` | Entity canonicalization map.            |
| `SPACY_MODEL`             | `en_core_web_sm`                       | spaCy model for query NER.                           |
| `TRINEDAY_TOP_K`          | `10`                                   | Top-K passages retrieved per query.                  |
| `HYBRID_ENTITY_WEIGHT`    | `4.0`                                  | Weight on the entity-FTS branch in score fusion.     |
| `HYBRID_FULLTEXT_WEIGHT`  | `1.0`                                  | Weight on the fulltext-FTS branch in score fusion.   |
| `HYBRID_ENTITY_LIMIT` / `HYBRID_FULLTEXT_LIMIT` | `200` / `200`            | Candidate-pool sizes for each FTS branch before fusion. |
| `MIN_GATE_SCORE_FLOOR`    | `17.0`                                 | BM25 floor below which the system declines to answer (when min-gating is enabled). |

## Bring your own corpus

The retrieval layer expects a SQLite database with FTS5 virtual tables matching the schema in `data/`. At minimum the schema includes:

- `chunks` — passage-level rows with `chunk_id`, `lookup_id`, source metadata, and full text
- `entity_fts` — FTS5 virtual table indexed over canonicalized entity names per chunk
- `fulltext_fts` — FTS5 virtual table indexed over the chunk text
- `entities`, `chunk_entities`, `topics`, `chunk_topics` — relational tables for entity and topic metadata

A corpus-prep pipeline lives under [`tools/`](tools/README.md). It chains EPUB → PDF → text → cleaned text → SQLite FTS5 DB. The headline tool is `tools/index/corpus_to_hybrid_db.py`, which takes a directory of `.txt` files and produces a DB matching the schema documented in `db/metadata.sql` and `db/load_metadata.py`. See `tools/README.md` for end-to-end recipes.

For sourcing the raw text in the first place, [`webscraper/`](webscraper/README.md) is a Playwright-capable crawler with per-domain YAML profiles and a resume-on-crash cache.

## Project structure

```
.
├── chat_server/            # The Flask web server (run from repo root with
│   │                       #   chat_server/ on PYTHONPATH; see scripts/run.sh)
│   ├── app.py              # App object, request hooks, startup wiring
│   ├── routes.py           # API endpoint handlers (single Flask blueprint)
│   ├── runtime.py          # Process-wide singletons (logger, rate limiter, ...)
│   ├── config.py           # Env-derived configuration, computed once at import
│   ├── auth.py             # Unlock gate, session + per-IP lockout, client-IP
│   ├── corpus.py           # Search / model-only / full RAG-chat orchestration
│   ├── state.py            # Shared RetrievalState, lazy init
│   ├── worker.py           # Background daemon thread draining the job queue
│   ├── email_responses.py  # Opt-in "email me the answer" for queued jobs
│   ├── rag_controller.py   # Model-adapter orchestration + in-process job queue
│   ├── rag_retrieval.py    # Hybrid SQLite FTS5 retrieval engine (algos 1–5)
│   ├── rag_context.py      # Prompt-context assembly from retrieved chunks
│   ├── model_adapters.py   # LLMStrategy ABC + five concrete backends
│   ├── model_prompts.py    # System prompt variants
│   ├── rag_cleaner.py      # Text and entity cleanup helpers
│   ├── db.py               # SQLite jobs + feedback (separate from corpus DB)
│   ├── utils.py            # Cross-cutting helpers (rate limiting, payload parsing)
│   ├── logging_config.py   # Logger setup
│   ├── gunicorn.conf.py    # Gunicorn config (production entry point)
│   ├── static/             # Frontend SPA — app.js, style.css, fonts, images
│   └── templates/          # Jinja templates (single-page app shell)
├── docker-compose.yml      # Container stack entry point
├── docs/                   # End-to-end documentation + hands-on codelab
├── deploy/                 # Dockerfile, .env.example, ops docs
├── data/                   # Corpus databases and entity-canon maps
├── db/                     # App database (jobs, feedback) and schema files
├── tools/                  # Corpus prep CLIs (chunking, cleaning, ingest)
├── train_ai/               # Optional LoRA fine-tuning scripts
├── webscraper/             # Playwright-capable crawler with per-domain profiles
├── eval/                   # Evaluation harness — labeler, A/B comparison
├── tests/                  # pytest suite (async chat, retrieval gate, adapters)
└── scripts/                # Deploy, eval, and ad-hoc test scripts
```

## Methodology and corpus

The reference corpus and default system prompts make editorial choices that are worth being explicit about. The codebase is corpus-agnostic — none of the retrieval or LLM-orchestration code is opinionated about subject matter — but the prompts shipped in `model_prompts.py` shape responses on contested topics by:

- Asking the model to classify a question as Established, Contested, Anomalous, or Low-evidence, rather than always defending a mainstream consensus.
- Surfacing competing claims when they appear in the retrieved evidence.
- Ending responses on contested topics with a "Plausibility Spectrum" (Strongly Supported / Moderately Supported / Indeterminate / Weakly Supported / Speculative / Disputed).

These prompt designs reflect a research-tool stance: the system is intended to surface evidence rather than enforce a single narrative. They are also configurable — operators can swap in their own prompts in `model_prompts.py` and select between them per request via `prompt_type` on `/api/chat`. See `model_prompts.py` for the full set of shipped variants.

## Project status

**This is alpha software.** It runs in production for an invited group of testers, but it is not yet hardened for the open internet. Known gaps being worked through:

- No CSRF protection on POST endpoints
- Per-session and per-IP unlock lockout now exist (`SOT_MAX_UNLOCK_ATTEMPTS`, `SOT_IP_LOCKOUT_SECS`), but the IP tracker is per-worker/in-memory — a shared store (Redis/DB) is still needed to make it robust under multiple gunicorn workers
- HuggingFace per-call timeout defaults are dev-friendly (900s); lower them in production (see config notes)
- Job queue persists in SQLite for inserts but is read in-process — losing in-flight jobs across worker restarts
- Recently added: a `pytest` suite under `tests/` (was shell-script integration tests only). CI to run it automatically is still pending.

Before deploying publicly, read [SECURITY.md](SECURITY.md) and address the gaps listed above.

## Lessons learned so far

The central lesson is that trustworthy AI does not come primarily from a smarter model. It comes from a disciplined system around the model: structured data, high-quality retrieval, source-grounded reasoning, validation, citation, careful prompting, and architecture that makes uncertainty and evidence visible.

- Data normalization and careful structuring is essential. For truth-seeking or evidence-heavy AI, the model is often less important than chunking, metadata quality, citation handling, search ranking, deduplication, and context assembly. A mediocre model with excellent retrieval can outperform a stronger model with messy retrieval.
- No model reliably retrieves a specific piece of training data on command. The model should not be trusted to remember or find facts internally. Search should retrieve evidence; the model should interpret, compare, summarize, and reason over that evidence.
- Censorship is baked into most models on a few levels: training data selection, alignment engineering, RLHF (Reinforcement Learning from Human Feedback), and platform/interface controls.
- Refusal ablation can remove alignment, but the loss of alignment/personality can destroy reasoning capabilities. This suggests that personality may play a crucial role in reasoning, perhaps providing a reasoning point of origin. Alignment, tone, values, caution, confidence, curiosity, and adversarial posture may not be superficial. They may influence what paths the model is willing or able to reason through. 
- AI reasoning follows a path and prompts that aren't structured to harness this may return suboptimal results. The same model can produce very different results depending on whether it is asked to summarize, investigate, challenge, compare, extract, or reason step by step. Prompt structure is not cosmetic; it determines the reasoning mode.
- It is possible to achieve virtually deterministic data transformation outputs by limiting the prompt to a single, well-defined task and employing rigorous output validation. A smaller model may fail at broad reasoning but perform well on narrow extraction, classification, formatting, validation, routing, summarization, or agent subroles.
- Cloud infrastructure for running larger models is prohibitively expensive. This suggests local-first infrastructure has strategic value. Local or edge inference gives more control over privacy, cost, censorship resistance, experimentation, and long-term independence.
- Metadata is intelligence. Dates, source names, authors, URLs, document type, topic tags, entities, credibility markers, and relationships between records are part of the system’s reasoning substrate.
- Multi-agent systems may be more powerful than single-model systems for controversial or complex topics. A single model tends to collapse toward one answer. Multiple specialized agents can retrieve, argue, critique, verify, and synthesize, making the interaction between models part of the intelligence.
- A RAG system can accidentally launder weak sources into authoritative answers. If bad, outdated, duplicated, or misleading sources enter the corpus, the AI may present them cleanly and confidently. Source quality controls matter.
- The user interface is part of the intelligence system. Mode selection, citations, source previews, search controls, history depth, confidence indicators, and comparison views can strongly affect response relevance and accuracy.
- Mode selection, citations, source previews, search controls, history depth, confidence indicators, and comparison views can strongly affect whether users understand and trust the output.
- Evaluation needs to be built into the pipeline from the beginning. You need test questions, known-answer checks, retrieval recall tests, citation accuracy checks, hallucination audits, and regression tests after every model, dataset, or prompt change.
- Training the same corpus on different base models can lead to surprising differences in model output.
- Training a LORA can take much longer than expected in terms of setup time, actual machine learning time, and necessary iterations in order to get good model output

## Roadmap

Near-term (path to v1.0):

- ~~`docker compose up` quickstart~~ ✓ landed — see [`deploy/README.md`](deploy/README.md). A small public sample corpus is still pending.
- ~~End-to-end documentation + hands-on codelab~~ ✓ landed — see [`docs/`](docs/README.md).
- ~~pytest suite~~ ✓ landed (`tests/`); GitHub Actions CI (ruff, mypy, pytest, pip-audit) still pending.
- ~~`CONFIG.md` reference~~ ✓ landed — see [`docs/CONFIG.md`](docs/CONFIG.md). A centralized configuration module (pydantic-settings) is still planned.
- Worker process split out of the Flask process; SQLite-backed queue as source of truth
- ~~Open-source the ingestion pipeline so users can build a corpus without reverse-engineering the schema~~ ✓ landed — see [`tools/README.md`](tools/README.md).
- Hardened security defaults (CSRF, rate-limit on auth endpoint, scrubbed logs)

Longer-term:

- Embeddings layer (currently BM25-only) for semantic recall on top of the lexical hybrid
- Multi-tenant deployment mode with per-user accounts (current model is a shared password gate)
- Web UI for corpus management and prompt editing

## Contributing

We welcome bug reports, feature suggestions, and pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test commands, and PR conventions.

## License

MIT License — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

Built on top of [spaCy](https://spacy.io/), [bm25s](https://github.com/xhluca/bm25s), [Flask](https://flask.palletsprojects.com/), [SQLite FTS5](https://www.sqlite.org/fts5.html), and the model providers integrated through `model_adapters.py`. The reference deployment uses corpora from independent journalism and primary-source archives, made available by their respective publishers.
