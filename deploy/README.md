# Seeds of Truth — Deployment

Container-based deployment for the Seeds of Truth Flask app. Single-host Docker Compose by default; the same image works behind nginx / Caddy / a managed PaaS.

## Contents of this directory

| File | Purpose |
| --- | --- |
| `Dockerfile` | Python 3.11-slim image with the app, gunicorn, spaCy model, and an unprivileged runtime user (UID 10001). Build context is the repo root. |
| `.env.example` | Template of every environment variable the app reads. Copy to `deploy/.env` and fill in before first run. |
| `README.md` | This file. |

The compose entry point lives one level up (`/docker-compose.yml`) so `docker-compose up` works from the repo root with no flags.

## First-time setup

```bash
# 1. Provide the env file (this is the only manual config step)
cp deploy/.env.example deploy/.env
$EDITOR deploy/.env
#   At minimum, set:
#     FLASK_SECRET_KEY  — `python -c 'import secrets; print(secrets.token_hex(32))'`
#     SOT_PASSWORDS     — comma-separated chat unlock passwords
#     One LLM backend's credentials (HF_API_KEY, DEEPINFRA_TOKEN, or SPARK_*)

# 2. Place your corpus DB
#    docker-compose.yml mounts ./data into the container at /app/data,
#    and HYBRID_DB_PATH is overridden to /app/data/gamma_master_hybrid_fts_stage3.db.
ls data/gamma_master_hybrid_fts_stage3.db   # confirm it exists

# 3. Bring the stack up
docker-compose up --build
```

App is then on `http://localhost:8000`. The first build takes 3–5 minutes (spaCy model download + lxml compile); subsequent builds reuse layers and finish in seconds.

## Day-to-day commands

```bash
# Start in the foreground (logs to terminal)
docker-compose up

# Start detached
docker-compose up -d

# Rebuild after a code change. The deps layer caches, so this is fast
# unless requirements.txt changed.
docker-compose up --build

# Tail logs
docker-compose logs -f web

# Stop containers (volumes preserved)
docker-compose down

# Stop AND delete the named volumes (DESTRUCTIVE — wipes the SQLite cache
# in db_cache/; data/ and logs/ are bind mounts and survive)
docker-compose down -v
```

## One-off commands inside the container

```bash
# Open a shell
docker-compose exec web bash

# Run an ingest into the mounted DB (writes go to the host's data/)
docker-compose exec web python tools/index/corpus_to_hybrid_db.py ingest \
    --corpus /app/data/my_corpus_txt \
    --db /app/data/my_corpus.db \
    --subset-name "My Source"

# Run a clean over the DB
docker-compose exec web python tools/clean/clean_chunks.py \
    --db /app/data/gamma_master_hybrid_fts_stage3.db --apply

# Smoke-test retrieval without leaving the container
docker-compose exec web python -c \
    "import rag_controller; print('boot OK; DB =', rag_controller.HYBRID_DB_PATH)"
```

If you need to share files into the container, drop them in `data/`, `logs/`, or `db_cache/` on the host — those are bind-mounted and visible inside `/app/data`, `/app/logs`, `/app/db_cache`.

## Health and monitoring

The Dockerfile defines a `HEALTHCHECK` that hits `http://127.0.0.1:8000/` every 30s. State surfaces in `docker-compose ps`:

```
NAME                 STATUS                    PORTS
seedsoftruth-web     Up 2 minutes (healthy)    0.0.0.0:8000->8000/tcp
```

If the status stays `(starting)` past the 30s grace period or flips to `(unhealthy)`, check logs (`docker-compose logs web`). Common causes are missing `FLASK_SECRET_KEY` (app crashes at boot) and missing DB at `HYBRID_DB_PATH` (boot fails the path-existence check in `rag_controller.py`).

Once a real `/healthz` endpoint lands in the app (tracked in `STACK_EVALUATION.md`), swap the Dockerfile's `HEALTHCHECK` to hit it.

## Troubleshooting

### Build fails on `pip install -r requirements.txt`

If pip dies on a wheel that needs compilation (typically `lxml` or `blis`/`spacy`), confirm the base image still has `build-essential`, `libxml2-dev`, `libxslt-dev` from the Dockerfile's `apt-get install` step. Slim Debian images sometimes drop these between minor versions; check the layer's stderr in the build log.

If the spaCy model URL fails to download, you're behind a proxy or the GitHub release has moved. Pin `en_core_web_sm` to a specific version in `requirements.txt` or download it inside the Dockerfile.

### Container starts then immediately exits

`docker-compose logs web` will show a Python traceback. The two most common causes:

1. **`FLASK_SECRET_KEY` not set** — the app refuses to boot when `FLASK_SECRET_KEY` is unset (outside `FLASK_DEBUG`) rather than fall back to an insecure default. Set it in `deploy/.env`.
2. **DB path doesn't resolve** — `rag_controller.py` boots by calling `HYBRID_DB_PATH.exists()` and raising `FileNotFoundError` if not. Confirm `data/gamma_master_hybrid_fts_stage3.db` exists on the host (the compose file overrides `HYBRID_DB_PATH` to the in-container mount path; you need the file on the host side of the bind mount).

### Healthcheck stays `(starting)` then flips `(unhealthy)`

The healthcheck hits `/`, which is served by Flask only after the worker thread has finished initial state load (spaCy model + entity-canonicalization map). On a cold start this can take 20–30 seconds. The Dockerfile gives it `--start-period=30s`. If you've added a large entity map and it's still not ready in time, bump the start period.

### Can reach `http://localhost:8000` but not `http://<host-ip>:8000` from another machine

`docker-compose.yml` publishes the port as `"8000:8000"`, which binds to `0.0.0.0` on the host by default. If the LAN can't reach you, the host firewall is in the way (`ufw`, `firewalld`, cloud VPC security group), not Docker.

If you wanted to bind only to localhost (for nginx-on-host to terminate TLS), change the port mapping to `"127.0.0.1:8000:8000"`.

### `database is locked` errors after a while

The in-process job queue + `workers = 1` keeps you safe for now, but if you've experimented with bumping `workers` in `gunicorn.conf.py` before the SQLite-backed queue lands (a planned change), you'll hit lock contention. Put it back to 1.

### Image is huge (~700 MB)

Mostly the spaCy model and PyTorch-flavor wheels pulled in transitively by `bm25s` / `sentence-transformers` if you've enabled topic classification. The `tools/index/corpus_to_hybrid_db.py --with-topics` path is heavy; if you don't use it, you can omit those packages from `requirements.txt` and shave 200+ MB.

A multi-stage build (`FROM ... AS builder` then `FROM python:3.11-slim AS runtime` with only `site-packages/` copied across) can cut another ~100 MB by dropping `build-essential` from the runtime image. Not landed yet.

### Bind-mounted `data/` is owned by `root` on the host after the container writes to it

The container runs as UID 10001 (the `sot` user). On Linux hosts where the bind-mount preserves UIDs, files created by the container show up as `10001:10001`, not root. If that's awkward, either `chown -R $(id -u):$(id -g) data/` periodically, or set `user: "${UID}:${GID}"` in `docker-compose.yml` and pass those vars at compose time.

On macOS / Docker Desktop, the file driver maps the container UID to your host user, so this generally isn't an issue.

## Production hardening checklist

The compose setup is good enough for a small invite-only alpha on a single VPS. Before opening to a wider audience:

- **TLS termination.** Put nginx, Caddy, or a managed load balancer in front of port 8000. The container should *not* terminate TLS itself.
- **Address the known alpha gaps.** Notably: split the in-process worker into its own process, cap `HF_TIMEOUT_SECS`, ensure `FLASK_SECRET_KEY` is set (the app now refuses to boot without it), and add a per-IP rate limit on `/api/unlock`.
- **Backups for `data/`.** The bind mount means the host filesystem is the source of truth. `tools/clean/clean_chunks.py` makes timestamped `.bak.<ts>` files but doesn't rotate them — add a cron job that prunes to last N.
- **Secrets management.** `deploy/.env` is fine for one machine. For multi-host or CI/CD, use Docker secrets, HashiCorp Vault, AWS Secrets Manager, or your platform's equivalent. Whatever you use, never bake secrets into the image.
- **Log rotation.** `logs/sot.log` is already 93 MB on the host. Configure `logrotate` on the host or switch `logging_config.py` to a rotating file handler.
- **Image vulnerability scanning.** `docker scout cves seedsoftruth:latest` after each build, or wire `trivy` / `grype` into CI.
- **Pin the base image by digest.** `FROM python:3.11-slim@sha256:...` instead of the floating tag, so a base-image rebuild can't silently shift behavior.

## Splitting the worker (future)

The web container currently runs the in-process queue worker as a daemon thread inside gunicorn. When the queue moves to SQLite-backed (a planned change), the architecture becomes:

```
       ┌────────────────┐         ┌────────────────┐
       │     web        │         │     worker     │
       │  (gunicorn)    │ ──────► │  (python -m    │
       │  /app/data ◄───┼──┐   ┌──┤    worker)     │
       └────────────────┘  │   │  └────────────────┘
                           ▼   ▼
                     ┌──────────────┐
                     │  data/*.db   │  ← bind mount on host
                     │   (SQLite)   │
                     └──────────────┘
```

The compose file has a commented-out `worker:` service stubbed at the bottom — uncomment it once you have `worker.py` (or `worker/__main__.py`) and remove the in-process thread from `app.py`. Both containers run from the same image, so `docker-compose up --build` rebuilds both in one shot.

## Deploying to a non-Compose target

The same Dockerfile works on:

- **Fly.io.** `fly launch` will detect it; pass `--dockerfile deploy/Dockerfile`.
- **Render.com.** Set "Dockerfile Path" to `deploy/Dockerfile` and "Docker Build Context Path" to `.`.
- **Kubernetes.** Build and push to a registry, then write a Deployment + Service + PersistentVolumeClaim for `/app/data`.
- **systemd directly on a VPS.** Build the image, then run with `docker run` in a systemd unit. Not as ergonomic as Compose but works.

For any of these: confirm the platform respects `HEALTHCHECK`, set the env file equivalent, and arrange persistent storage for `/app/data`.
