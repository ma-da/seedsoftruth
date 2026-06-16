# syntax=docker/dockerfile:1.6
#
# Seeds of Truth — Flask + gunicorn container image.
#
# Build from the repo root:
#     docker build -f deploy/Dockerfile -t seedsoftruth:latest .
#
# Or (recommended) via docker-compose, which sets the file path for you:
#     docker-compose up --build
#
# Notes:
#   * The build context is the REPO ROOT, not deploy/. The Dockerfile only
#     lives in deploy/ for organization; COPY paths below are relative to
#     the root.
#   * data/ is NOT copied in. It's mounted as a volume by docker-compose so
#     the 1.7 GB SQLite DB persists across container rebuilds.
#   * spaCy's en_core_web_sm model is installed from a GitHub release URL
#     listed in requirements.txt — no separate `python -m spacy download`
#     step needed.
#   * The image runs as a non-root user (UID 10001) for least-privilege.

# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages:
#   build-essential, libxml2-dev, libxslt-dev — needed by some Python
#   wheels (lxml, blis/spacy on platforms without prebuilt wheels).
#   ca-certificates — TLS for outbound calls to HF / DeepInfra / Spark.
#   curl — used by the healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt-dev \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies (cached layer)
# ---------------------------------------------------------------------------
# Copying requirements.txt first means this layer only invalidates when
# deps change, not every time application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
# Explicit COPY of code files (rather than `COPY . .`) keeps the image
# minimal and avoids accidentally baking in data/, logs/, .venv/, etc.
# .dockerignore at the repo root provides defense-in-depth.
# The Flask web server (all modules + templates/ + static/) lives under
# chat_server/. Copy the whole package in one shot — this also guarantees
# every module the app imports is present (the old explicit file list was
# incomplete). gunicorn.conf.py adds chat_server/ to the import path so the
# flat imports and the `app:app` WSGI target resolve from /app.
COPY chat_server/ ./chat_server/

COPY tools/      ./tools/
COPY webscraper/ ./webscraper/

# Create the directories that will receive bind mounts so the entrypoint
# doesn't fail if a fresh `docker run` is invoked without volumes.
RUN mkdir -p /app/data /app/logs /app/db_cache

# ---------------------------------------------------------------------------
# Non-root user
# ---------------------------------------------------------------------------
RUN groupadd --system --gid 10001 sot \
 && useradd --system --uid 10001 --gid sot --no-create-home --shell /sbin/nologin sot \
 && chown -R sot:sot /app

USER sot

# ---------------------------------------------------------------------------
# Network + runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

# Healthcheck — replace `/` with `/healthz` once you add the endpoint
# (see STACK_EVALUATION.md "Critical gaps").
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent --output /dev/null http://127.0.0.1:8000/ || exit 1

# Bind to 0.0.0.0 inside the container so the host (or a sibling container
# like nginx) can reach gunicorn. The existing gunicorn.conf.py binds to
# 127.0.0.1, which is correct for bare-metal but wrong for containers; we
# override here without modifying the config file used outside Docker.
CMD ["gunicorn", "-c", "chat_server/gunicorn.conf.py", "-b", "0.0.0.0:8000", "app:app"]
