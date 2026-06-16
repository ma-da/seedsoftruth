"""Seeds of Truth Flask app — application object, request hooks, wiring.

Builds the Flask ``app``, registers the API blueprint from :mod:`routes`,
installs the request hooks (time gate, no-cache, debug logging), and
performs one-time startup: retrieval-state boot, database init, and the
background worker thread. Route handlers live in :mod:`routes`; shared
singletons and helpers live in :mod:`runtime`.
"""

from __future__ import annotations

import os
import platform
import secrets

from flask import Flask, abort, request

import config
import db
import logging_config
import routes
import runtime
import state
import worker
from rag_controller import ENABLE_MIN_GATING

app_logger = logging_config.get_logger("app")

# ------------------ App setup ------------------

app = Flask(__name__)

# Session signing key. Required in production: set via systemd override,
#   Environment="FLASK_SECRET_KEY=...long random hex..."
# (generate with: python -c 'import secrets; print(secrets.token_hex(32))').
# Outside debug mode we refuse to start without it rather than fall back to
# a known, forgeable default. Under FLASK_DEBUG we mint an ephemeral random
# key so local dev just works (sessions reset on restart).
_secret_key = os.environ.get("FLASK_SECRET_KEY")
if not _secret_key:
    _flask_debug = (os.environ.get("FLASK_DEBUG", "") or "").strip().lower()
    if _flask_debug in ("1", "true", "yes", "on"):
        _secret_key = secrets.token_hex(32)
        app_logger.warning(
            "FLASK_SECRET_KEY is unset; generated an ephemeral key for this "
            "debug session. Sessions will not survive a restart. Set "
            "FLASK_SECRET_KEY for stable sessions."
        )
    else:
        raise RuntimeError(
            "FLASK_SECRET_KEY is not set. Refusing to start with an insecure "
            "default. Set FLASK_SECRET_KEY to a long random hex value "
            "(python -c 'import secrets; print(secrets.token_hex(32))'), or "
            "run with FLASK_DEBUG=1 for an ephemeral local-dev key."
        )
app.secret_key = _secret_key

# Dev-style template reloading (turn off in prod)
if os.environ.get("TEMPLATES_AUTO_RELOAD", "1") == "1":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

app.register_blueprint(routes.bp)


# ------------------ Request hooks ------------------


@app.before_request
def time_gate():
    """before_request hook: enforce the business-hours time gate.

    Handlers decorated with ``runtime.no_time_gate`` are exempt.
    """
    endpoint = app.view_functions.get(request.endpoint)
    if getattr(endpoint, "_no_time_gate", False):
        return

    if not runtime.is_within_service_hours():
        abort(
            403,
            description=f"API available {config.START_TIME.strftime('%I:%M %p').lstrip('0').lower()}–"
            f"{config.END_TIME.strftime('%I:%M %p').lstrip('0').lower()} EST",
        )


@app.after_request
def no_cache_html(resp):
    """after_request hook: disable browser caching of HTML responses.

    Args:
        resp: The outgoing Flask response.

    Returns:
        The response, with no-cache headers added when its mimetype is
        ``text/html``.
    """
    # Prevent caching for HTML during dev
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


@app.before_request
def debug_request() -> None:
    """before_request hook: log method, path, and body of /api/* requests.

    A development aid for inspecting exact browser payloads. Reads the
    body with ``cache=True`` so later ``get_json()`` calls still work.
    """
    raw = request.get_data(cache=True)  # cache=True keeps get_json() working
    if request.path.startswith("/api/"):
        runtime.app_logger.info("--- %s %s ---", request.method, request.path)
        if raw:
            runtime.app_logger.info("Body (first 2000 bytes): %s", raw[:2000])
        runtime.app_logger.info("JSON: %s", request.get_json(silent=True))


# ------------------ Startup ------------------

# Eager init once at import time (fine under systemd + gunicorn). If this
# fails (network down, etc.), ensure_state() keeps retrying on requests.
state.init_state(force=False)
db.init_db()

# Sweep any jobs left in 'queued' or 'processing' from before a restart.
# The in-memory queue doesn't persist across gunicorn restarts, so EVERY
# row in those states is orphaned by definition — there is no live worker
# that could ever pick them up. We use stale_minutes=0 (sweep all) rather
# than stale_minutes=10 because the 10-minute grace period only makes
# sense for a live process where in-flight rows might still be working
# their way through the queue. At startup, "in-flight" is impossible:
# the queue is empty.
# Best-effort: if SQLite is unavailable, log and continue; the rest of
# the app shouldn't fail to boot over a sweep miss.
try:
    db.sweep_orphaned_jobs(stale_minutes=0)
except Exception:
    runtime.app_logger.exception(
        "startup sweep_orphaned_jobs failed (best-effort, swallowed)"
    )

worker.init_worker()

if ENABLE_MIN_GATING:
    runtime.app_logger.info("Min gating feature is enabled")
else:
    runtime.app_logger.info("Min gating feature is disabled")

runtime.app_logger.info(f"Python version: {platform.python_version()}")


# ------------------ Local dev runner ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
