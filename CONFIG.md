# Configuration Reference

All runtime configuration is read from environment variables. This document is the authoritative reference; the table in the README is a minimal subset for quickstart purposes.

> **Roadmap note:** Once Phase 1 of the v1.0 cleanup lands, this will be replaced by a generated reference produced from a `pydantic-settings` configuration class. Until then, this file is maintained by hand and verified against the source on each release.

## Required

| Variable           | Default                | Read in        | Description                                                                                              |
|--------------------|------------------------|----------------|----------------------------------------------------------------------------------------------------------|
| `FLASK_SECRET_KEY` | *(required, no default)* | `app.py`       | Flask session-signing key. **Set this in production** — the in-code fallback (`"dev-change-me"`) is a known issue and is being removed. Use a long random hex value (e.g., `python -c "import secrets; print(secrets.token_hex(32))"`). |
| `SOT_PASSWORDS`    | `""` (empty)             | `app.py`       | Comma-separated allow-list of passwords for the chat password gate (`/api/unlock`). If empty, no one can unlock chat. The search endpoint remains open regardless. |

## LLM backend selection

| Variable        | Default | Read in              | Description                                                                                  |
|-----------------|---------|----------------------|----------------------------------------------------------------------------------------------|
| `MODEL_ADAPTER` | `hf`    | `rag_controller.py`  | Which LLM backend to use as the default. One of: `hf` (HuggingFace), `deepinfra`, `spark` (Cloudflare-Access internal endpoint), `sim` (deterministic stub for testing). Per-request override is available on `/api/chat` via the `model_type` field. |

## LLM — HuggingFace (`hf`)

| Variable           | Default | Read in              | Description                                                                                                       |
|--------------------|---------|----------------------|-------------------------------------------------------------------------------------------------------------------|
| `HF_API_KEY`       | `""`    | `model_adapters.py`  | HuggingFace Inference Endpoints API key.                                                                          |
| `HF_TIMEOUT_SECS`  | `900`   | `model_adapters.py`  | Per-LLM-call HTTP timeout in seconds. **Recommended override: 120 in production.** The 900s default is a development convenience that becomes a denial-of-service vector when exposed publicly. |
| `HF_MAX_ATTEMPTS`  | `10`    | `model_adapters.py`  | Number of retries on HTTP 503 (HuggingFace endpoint cold start). **Recommended override: 3 in production.**       |
| `HF_MAX_WAIT_SECS` | `6`     | `model_adapters.py`  | Maximum wait between retries when HuggingFace returns an `estimated_time` hint.                                   |

The HuggingFace Inference Endpoint URL is currently hardcoded in `model_adapters.py:HF_ENDPOINT_URL`. Promoting it to an environment variable is on the v1.0 roadmap.

## LLM — DeepInfra (`deepinfra`)

| Variable             | Default                                  | Read in              | Description                                                                                                |
|----------------------|------------------------------------------|----------------------|------------------------------------------------------------------------------------------------------------|
| `DEEPINFRA_TOKEN`    | `""`                                       | `model_adapters.py`  | API token for DeepInfra inference. Used by the web app at runtime.                                          |
| `DEEPINFRA_API_KEY`  | `""`                                       | `eval/labeler.py`    | Same credential, but read by the eval harness under a different name. **Known inconsistency** — set both to the same value or unify them in your environment. Unifying these is on the v1.0 roadmap. |
| `DEEPINFRA_MODEL`    | `meta-llama/Meta-Llama-3.1-70B-Instruct` | `model_adapters.py`  | DeepInfra model identifier.                                                                                |

## LLM — Spark (Cloudflare-Access internal endpoint)

This adapter is used to call an internal endpoint protected by Cloudflare Access. Most operators will not need it.

| Variable                       | Default                              | Read in              | Description                                                  |
|--------------------------------|--------------------------------------|----------------------|--------------------------------------------------------------|
| `SPARK_BASE_URL`               | `https://seedsoftruth.peerservice.org` | `model_adapters.py`  | Base URL of the Spark service.                               |
| `SPARK_SITE_API_KEY`           | `""`                                   | `model_adapters.py`  | Site-level API key for the Spark endpoint.                   |
| `SPARK_CF_ACCESS_CLIENT_ID`    | `""`                                   | `model_adapters.py`  | Cloudflare Access service-token client ID.                   |
| `SPARK_CF_ACCESS_CLIENT_SECRET`| `""`                                   | `model_adapters.py`  | Cloudflare Access service-token client secret.               |
| `SPARK_MODEL_NAME`             | `wtk_gamma_v9`                         | `model_adapters.py`  | Model identifier passed through to the Spark service.        |

## LLM — Sim (test simulator)

The `sim` adapter requires no environment variables. It returns deterministic stub responses without making any network calls. Use it for retrieval-only development and integration tests.

## Retrieval

| Variable                  | Default                                                   | Read in              | Description                                                                                              |
|---------------------------|-----------------------------------------------------------|----------------------|----------------------------------------------------------------------------------------------------------|
| `HYBRID_DB_PATH`          | `./data/gamma_master_hybrid_fts_stage2.db`                | `rag_controller.py`  | Path to the corpus FTS5 SQLite database. See README → "Bring your own corpus" for the expected schema.   |
| `ENTITY_CANON_MAP_PATH`   | `./data/entity_query_normalization_map.flat.json`         | `rag_controller.py`  | Path to the entity-canonicalization map (maps surface forms to canonical entity terms used in the index). |
| `SPACY_MODEL`             | `en_core_web_sm`                                          | `rag_controller.py`  | spaCy model used for query-time NER. Must be installed and loadable.                                     |
| `RETRIEVAL_BACKEND`       | `sqlite_hybrid`                                           | `rag_controller.py`  | Selects the retrieval backend. Currently only `sqlite_hybrid` is supported; the variable exists for future backend additions. |
| `HYBRID_ENTITY_WEIGHT`    | `4.0`                                                     | `rag_controller.py`  | Weight on the entity-FTS branch in the score-fusion formula.                                             |
| `HYBRID_FULLTEXT_WEIGHT`  | `1.0`                                                     | `rag_controller.py`  | Weight on the fulltext-FTS branch in the score-fusion formula.                                           |
| `HYBRID_ENTITY_LIMIT`     | `200`                                                     | `rag_controller.py`  | Maximum number of candidate chunks pulled from the entity-FTS branch before fusion.                      |
| `HYBRID_FULLTEXT_LIMIT`   | `200`                                                     | `rag_controller.py`  | Maximum number of candidate chunks pulled from the fulltext-FTS branch before fusion.                    |
| `TRINEDAY_TOP_K`          | `10`                                                      | `rag_controller.py`  | Default `top_k` for retrieval queries when no value is supplied. **Note:** this name is a legacy artifact from when the project was scoped to a single corpus; renaming to `DEFAULT_TOP_K` is on the v1.0 roadmap. |
| `MIN_GATE_SCORE_FLOOR`    | `17.0`                                                    | `rag_controller.py`  | BM25 score floor below which the system declines to answer (when min-gating is enabled). Calibrated empirically against the in-domain test set; see the docstring on the constant for details. |

## Operational

| Variable                   | Default | Read in              | Description                                                                                                 |
|----------------------------|---------|----------------------|-------------------------------------------------------------------------------------------------------------|
| `TEMPLATES_AUTO_RELOAD`    | `1`     | `app.py`             | If set to `1`, Jinja templates auto-reload on change. Set to `0` in production.                             |
| `SOT_INIT_RETRY_COOLDOWN_S`| `10`    | `app.py`             | Cooldown (seconds) between retries of the retrieval-state boot when initial load fails.                     |

## Eval harness

These variables are read only by code under `eval/` and `scripts/`. They are not needed to run the web app.

| Variable             | Default | Read in            | Description                                                                                                                          |
|----------------------|---------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `ANTHROPIC_API_KEY`  | `""`    | `eval/labeler.py`  | Anthropic API key, used when running the labeler with `--judge anthropic`.                                                            |
| `DEEPINFRA_API_KEY`  | `""`    | `eval/labeler.py`, `scripts/bootstrap_eval.sh` | DeepInfra API key, used when running the labeler with `--judge deepinfra`. See note above about the `DEEPINFRA_TOKEN` / `DEEPINFRA_API_KEY` inconsistency. |
| `BASE_URL`           | `http://localhost:5000` | `scripts/run_eval.sh`, `scripts/bootstrap_eval.sh` | Base URL of the running Flask app. The eval harness hits its `/api/search` endpoint. |

## Tools (`tools/prober.py`)

These variables are read only by the standalone diagnostic tool `tools/prober.py`. They do not affect the running web app.

| Variable        | Default                  | Description                                                  |
|-----------------|--------------------------|--------------------------------------------------------------|
| `SOT_URL`       | `http://localhost:5000`  | Base URL of the service to probe.                            |
| `SOT_PASSWORD`  | *(unset)*                | Password for the prober's auth flow. Note this is **singular** — distinct from the web app's `SOT_PASSWORDS` (plural). |
| `NO_COLOR`      | *(unset)*                | If set, disables color output in prober logs.                |

## Production checklist

Minimum environment for a defensible production deployment:

```bash
# Required
export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export SOT_PASSWORDS="<long-rotated-shared-password>"

# Pick one backend
export MODEL_ADAPTER="deepinfra"
export DEEPINFRA_TOKEN="<token>"
export DEEPINFRA_API_KEY="<token>"   # same value, set both for now (see DeepInfra section)

# Production-safe LLM call bounds
export HF_TIMEOUT_SECS=120
export HF_MAX_ATTEMPTS=3

# Corpus
export HYBRID_DB_PATH=/var/lib/seedsoftruth/corpus.db
export ENTITY_CANON_MAP_PATH=/var/lib/seedsoftruth/entity_canon_map.flat.json

# Operational
export TEMPLATES_AUTO_RELOAD=0
```

Run under a process supervisor (systemd, supervisord) with restart-on-failure, log rotation, and a memory ceiling. See `scripts/deploy_web.sh` for the reference deployment pattern.
