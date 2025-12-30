"""
Seeds of Truth Flask App

Integrates rag_controller.py:
- boot() builds immutable RetrievalState (downloads registry + artifacts) :contentReference[oaicite:3]{index=3}
- search_references(...) returns retrieval-only JSON results :contentReference[oaicite:4]{index=4}
- ask(...) does retrieval + calls LLM proxy FLASK_PROXY_URL :contentReference[oaicite:5]{index=5}

Endpoints:
- GET  /                  -> templates/index.html
- POST /api/ping           -> demo ping
- POST /api/unlock         -> password sets session gate
- GET  /api/access         -> access state
- POST /api/search         -> ALWAYS allowed (retrieval-only)
- POST /api/chat           -> gated, uses rag_controller.ask(...)
- POST /api/ab             -> gated, calls ask twice (simple A/B dev)
- GET  /api/status         -> health
- GET  /api/queue          -> dev queue stub

Notes:
- This initializes retrieval_state once per gunicorn worker process.
- boot() and retrieval call out to the network (Hive + JSON artifacts), so startup can be slow.
- For dev reliability, ensure_state() lazily retries initialization.
"""

from __future__ import annotations

import os
import hmac
import time
import threading
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import rag_controller

from asgiref.sync import async_to_sync  # pip install asgiref
from flask import Flask, render_template, request, jsonify, session

import traceback
import inspect
import utils

import db


# ------------------ Logging ------------------

def _setup_logger() -> logging.Logger:
    try:
        import logging_config  # type: ignore
        return logging_config.setup_logging(logging.INFO)
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        return logging.getLogger("seedsoftruth")


app_logger = _setup_logger()

# ------------------ App setup ------------------

app = Flask(__name__)

# Set via systemd override:
#   Environment="FLASK_SECRET_KEY=...long random..."
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-me")

# Dev-style template reloading (turn off in prod)
if os.environ.get("TEMPLATES_AUTO_RELOAD", "1") == "1":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

# Due to GIL, booleans should be thread-safe.
app_is_shutdown = False

# inflight requests
inflight_chat_reqs = utils.SafeInt(0)

# User rate limiter
RATE_LIMITING_INTERVAL = 30
rate_limiter = utils.SimpleUserRateLimiter(RATE_LIMITING_INTERVAL)

# ------------------ Password gating ------------------

def _parse_passwords(env_value: str) -> List[str]:
    return [p.strip() for p in (env_value or "").split(",") if p.strip()]


ALLOWED_PASSWORDS = _parse_passwords(os.environ.get("SOT_PASSWORDS", ""))


def _is_unlocked() -> bool:
    return bool(session.get("unlocked", False))


def _require_unlocked():
    if not _is_unlocked():
        return jsonify({"ok": False, "error": "locked", "message": "Not today"}), 403
    return None


# ------------------ Async bridge helpers ------------------

def _async_to_sync(async_fn):
    """
    Convert an async function (no args) to sync call.
    Prefers asgiref, which is the most reliable for sync Flask under gunicorn.
    """
    try:
        return async_to_sync(async_fn)
    except Exception as e:
        app_logger.error("Exception path in _async_to_sync, details: {e}")
        # fallback: run in a fresh event loop (ok for dev; not ideal at scale)
        import asyncio
        def _runner():
            return asyncio.run(async_fn())

        return _runner


# ------------------ Retrieval state (boot + ensure) ------------------

retrieval_state = None
_state_lock = threading.Lock()
_last_init_attempt_ts = 0.0
_last_init_error: Optional[str] = None

INIT_RETRY_COOLDOWN_S = int(os.environ.get("SOT_INIT_RETRY_COOLDOWN_S", "10"))


def init_state(force: bool = False) -> bool:
    """
    Initialize retrieval_state by calling rag_controller.boot().

    - Uses a lock to avoid concurrent boot calls.
    - Uses a cooldown to avoid spamming boot if it fails.
    - Returns True if retrieval_state is ready.
    """
    global retrieval_state, _last_init_attempt_ts, _last_init_error

    if retrieval_state is not None:
        return True

    now = time.time()
    if (not force) and (now - _last_init_attempt_ts) < INIT_RETRY_COOLDOWN_S:
        return False

    with _state_lock:
        if retrieval_state is not None:
            return True

        now = time.time()
        if (not force) and (now - _last_init_attempt_ts) < INIT_RETRY_COOLDOWN_S:
            return False

        _last_init_attempt_ts = now
        _last_init_error = None

        app_logger.info("init_state: starting rag_controller.boot()...")

        try:
            retrieval_state = rag_controller.boot()
            if retrieval_state is None:
                _last_init_error = "boot() returned None"
                app_logger.error(_last_init_error)
                return False

            app_logger.info("init_state: ✅ retrieval_state initialized")
            return True

        except Exception as e:
            _last_init_error = f"boot() failed: {e}"
            app_logger.exception(_last_init_error)
            retrieval_state = None
            return False


def worker_body():
    global app_is_shutdown
    MODEL_NOT_READY_REPORTING_INTERVAL = 5
    WORKER_NO_WORK_SLEEP_INTERVAL_SECS = 5
    WORKER_NO_QUEUED_JOB_REPORTING_INTERVAL = 20

    model_not_ready_interval = MODEL_NOT_READY_REPORTING_INTERVAL - 1  # let's first report happen sooner
    worker_no_queued_job_interval = WORKER_NO_QUEUED_JOB_REPORTING_INTERVAL - 1

    worker_ready_report = False
    app_logger.info("App worker started")

    while not app_is_shutdown:
        if not rag_controller.has_queued_job():
            worker_no_queued_job_interval = worker_no_queued_job_interval + 1
            if worker_no_queued_job_interval > WORKER_NO_QUEUED_JOB_REPORTING_INTERVAL:
                app_logger.info("Worker thread heartbeat. No job work to do.")
                worker_no_queued_job_interval = 0

            time.sleep(WORKER_NO_WORK_SLEEP_INTERVAL_SECS)
            continue
        else:
            worker_no_queued_job_interval = 0

        model_ready = _async_to_sync(rag_controller.is_model_ready)()
        if not model_ready:
            model_not_ready_interval = model_not_ready_interval + 1
            worker_ready_report = False

            if model_not_ready_interval > MODEL_NOT_READY_REPORTING_INTERVAL:
                app_logger.info("Model still not ready. Worker body waiting.")
                model_not_ready_interval = 0

            time.sleep(WORKER_NO_WORK_SLEEP_INTERVAL_SECS)
            continue
        else:
            model_not_ready_interval = 0

        if not worker_ready_report:
            app_logger.info("Model is ready. Worker is active.")
            worker_ready_report = True

        queued_job = rag_controller.get_next_queued_job()
        if not queued_job:
            time.sleep(WORKER_NO_WORK_SLEEP_INTERVAL_SECS)
            continue

        try:
            app_logger.info(f"Worker chat_with_corpus for job_id {queued_job.job_id} with user_id {queued_job.user_id}...")
            answer, docs = chat_with_corpus(queued_job.prompt, top_k=10)

            resp = rag_controller.QueuedResponse(
                ok=True,
                error="none",
                detail="success",
                job_id=queued_job.job_id,
                user_id=queued_job.user_id,
                prompt=queued_job.prompt,
                reply=answer,
                references=docs
            )
            rag_controller.queue_outgoing(resp)

            app_logger.info(f"Worker query job_id {queued_job.job_id} was executed succesfully. Marking done in db.")
            db.mark_done(queued_job.job_id, answer)

        except RuntimeError as e:
            app_logger.error("chat_with_corpus failed in worker with runtime error: {e}")

            resp = rag_controller.QueuedResponse(
                ok=False,
                error="Search system not initialized",
                detail=str(e),
                job_id=queued_job.job_id,
                user_id=queued_job.user_id,
                prompt=queued_job.prompt,
                reply="none",
                references="none"
            )
            rag_controller.queue_outgoing(resp)

        except Exception as e:
            app_logger.error("chat_with_corpus failed in worker with exception: {e}")

            resp = rag_controller.QueuedResponse(
                ok=False,
                error="Chat failed",
                detail=str(e),
                job_id=queued_job.job_id,
                user_id=queued_job.user_id,
                prompt=queued_job.prompt,
                reply="none",
                references="none"
            )
            rag_controller.queue_outgoing(resp)

        rag_controller.pop_next_queued_job()

    app_logger.info("App worker shutdown")


def init_worker():
    worker = threading.Thread(target=worker_body)
    worker.daemon = True
    worker.start()
    logging.info("Started daemon worker thread")


def ensure_state() -> bool:
    """Lazy initializer used by routes. Returns True if ready."""
    return init_state(force=False)


# Eager init once at import time (fine under systemd + gunicorn).
# If this fails (network down, etc.), ensure_state() will keep retrying on requests.
init_state(force=False)
db.init_db()
init_worker()


# ------------------ Shared search/chat functions ------------------

def search_corpus(query: str, top_k: int, shard_k: int = 20) -> Dict[str, Any]:
    """
    Retrieval-only search using rag_controller.search_references(state,...).

    Mirrors the known-good JS:
      - centroid routing to shard_k shards
      - full scoring on those shards
      - returns top_k results
    """
    global retrieval_state

    if not ensure_state():
        app_logger.error("Search system not initialized. Did boot() work?")
        raise RuntimeError(_last_init_error or "Search system not initialized")

    import rag_controller  # type: ignore

    async def _run():
        return await rag_controller.search_references(  # type: ignore
            retrieval_state,
            query,
            top_k=int(top_k),
            centroid_k=int(shard_k),
            verbose=False,
        )

    out = _async_to_sync(_run)() or {}
    if not isinstance(out, dict):
        return {"results": [], "num_results": 0, "query": query}

    # Ensure JSON-safe
    results = out.get("results", [])
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                # cast numpy scalars if any leaked through
                v = r.get("score_tfidf")
                if v is not None:
                    try:
                        r["score_tfidf"] = float(v)
                    except Exception:
                        pass
                v = r.get("score")
                if v is not None:
                    try:
                        r["score"] = float(v)
                    except Exception:
                        pass

    return out


def ask_corpus(question: str) -> str:
    """
    Retrieval + LLM proxy answer using rag_controller.ask(state,...).
    """
    global retrieval_state

    if not ensure_state():
        raise RuntimeError(_last_init_error or "Search system not initialized")

    import rag_controller  # type: ignore

    async def _run():
        return await rag_controller.ask(  # type: ignore
            retrieval_state,
            question,
            verbose=False,
        )

    ans = _async_to_sync(_run)()
    return (ans or "").strip()


def chat_with_corpus(query: str, top_k: int = 10):
    """
    Returns (answer: str, docs: list[dict])

    This bridges async rag_controller functions into sync Flask code.

    Assumptions / compatibility:
    - Preferred: rag_controller.ask(state, question, verbose=...) -> str
      (ask does retrieval + context + LLM)
    - Also tries to fetch docs for UI references via:
        - rag_controller.search_references(state, query, top_k=...)
          OR rag_controller.sparse_retrieve(state, query)
    """
    global retrieval_state
    if retrieval_state is None:
        raise RuntimeError("Search system not initialized")

    q = (query or "").strip()
    if not q:
        return ("", [])

    async def _run():
        # ---------- 1) Get answer from LLM ----------
        answer = await rag_controller.ask(retrieval_state, q, verbose=False)

        # ---------- 2) Get docs for references ----------
        # Prefer a dedicated search_references() if present, since it often returns richer metadata
        docs = []
        pack = await rag_controller.search_references(retrieval_state, q, top_k=top_k)
        if isinstance(pack, dict):
            docs = pack.get("results", []) or []
        elif isinstance(pack, list):
            docs = pack

        # TODO: Old code. Remove when no longer needed.
        # docs = await rag_controller.sparse_retrieve(retrieval_state, q)

        # normalize output types
        if not isinstance(docs, list):
            docs = []

        return (str(answer or "").strip(), docs[: max(0, int(top_k) or 0)])

    return async_to_sync(_run)()


# ------------------ Routes: UI ------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.after_request
def no_cache_html(resp):
    # Prevent caching for HTML during dev
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


# ------------------ Routes: Debug request logging (optional but useful) ------------------

@app.before_request
def debug_request():
    # Comment out if too noisy. Helpful for seeing EXACT browser payloads.
    raw = request.get_data(cache=True)  # cache=True keeps get_json() working
    if request.path.startswith("/api/"):
        app_logger.info("--- %s %s ---", request.method, request.path)
        app_logger.info("Content-Type: %s", request.headers.get("Content-Type"))
        if raw:
            app_logger.info("Body (first 2000 bytes): %s", raw[:2000])
        app_logger.info("JSON: %s", request.get_json(silent=True))


# ------------------ Routes: Auth ------------------

@app.post("/api/unlock")
def api_unlock():
    payload = request.get_json(silent=True) or {}
    pw = (payload.get("password") or "").strip()

    if not pw or not ALLOWED_PASSWORDS:
        session["unlocked"] = False
        return jsonify({"ok": False, "message": "Not today"}), 403

    # each unlock creates a new uuid
    # TODO: Make this robust against an attacker who farms uuids.
    new_uuid = uuid.uuid4()

    ok = any(hmac.compare_digest(pw, real) for real in ALLOWED_PASSWORDS)
    session["unlocked"] = bool(ok)

    return (jsonify({"ok": True,
                     "message": "Access Granted",
                     "user_id": str(new_uuid),
                     }), 200) if ok else \
        (jsonify({"ok": False, "message": "Not today"}), 403)


@app.get("/api/access")
def api_access():
    return jsonify({"ok": True, "unlocked": _is_unlocked()}), 200


# ------------------ Routes: Demo ping ------------------

@app.post("/api/ping")
def api_ping():
    data = request.get_json(force=True)
    return jsonify({
        "ok": True,
        "received": data,
        "server_time": datetime.utcnow().isoformat() + "Z",
        "message": "Flask received your message successfully."
    })


# ------------------ Routes: Search (always allowed) ------------------

@app.post("/api/search")
def api_search():
    try:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()

        # Back-compat: existing UI sends max_n; new UI can send top_k
        try:
            top_k = int(payload.get("top_k", payload.get("max_n", 10)))
        except Exception:
            return jsonify({"ok": False, "error": "Field 'top_k'/'max_n' must be an integer"}), 400

        # Closest shards to search (centroid routing count)
        try:
            shard_k = int(payload.get("shard_k", 20))
        except Exception:
            return jsonify({"ok": False, "error": "Field 'shard_k' must be an integer"}), 400

        if not query:
            return jsonify({"ok": False, "error": "Field 'query' must be a non-empty string"}), 400
        if top_k <= 0 or top_k > 200:
            return jsonify({"ok": False, "error": "Field 'top_k' must be between 1 and 200"}), 400
        if shard_k <= 0 or shard_k > 200:
            return jsonify({"ok": False, "error": "Field 'shard_k' must be between 1 and 200"}), 400

        # Call search_corpus in a compatible way
        sig = inspect.signature(search_corpus)
        params = sig.parameters

        if "shard_k" in params:
            results = search_corpus(query, top_k=top_k, shard_k=shard_k)
        elif "centroid_k" in params:
            results = search_corpus(query, top_k=top_k, centroid_k=shard_k)  # in case you used centroid_k
        else:
            # old signature: search_corpus(query, top_k)
            results = search_corpus(query, top_k)

        # Normalize results BEFORE .get calls (prevents HTML 500)
        if not isinstance(results, dict):
            results = {"results": []}

        out_list = results.get("results", [])
        if not isinstance(out_list, list):
            out_list = []

        num_results = results.get("num_results")
        if not isinstance(num_results, int):
            num_results = len(out_list)

        message = results.get("message")
        if not isinstance(message, str) or not message.strip():
            message = f"Found {num_results} result(s)."

        return jsonify({
            "ok": True,
            "query": query,
            "num_results": num_results,
            "message": message,
            "references": out_list,
            "results": out_list,
            "top_k": top_k,
            "shard_k": shard_k,
        }), 200

    except Exception as e:
        # Always return JSON so the frontend sees the real error.
        tb = traceback.format_exc()
        try:
            app_logger.exception("Search failed")
        except Exception:
            pass

        return jsonify({
            "ok": False,
            "error": "Search failed",
            "detail": str(e),
            "traceback": tb,
        }), 500


# ------------------ Routes: Chat (gated) ------------------

@app.post("/api/chat")
def api_chat():
    global inflight_chat_reqs, rate_limiter

    locked = _require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    msg = (payload.get("message") or payload.get("query") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "Field 'message' must be a non-empty string"}), 400

    # optional param that lets us test force to the queue
    force_queue_str = (payload.get("force_queue") or "").strip()
    force_queue = utils.str_to_bool(force_queue_str, strict=False)

    user_id = payload.get("user_id", "none")
    if not isinstance(user_id, str):
        app_logger.warn("Chat request user_id was invalid")
        return jsonify({
            "ok": False,
            "error": "Field 'user_id' must be a string"
        }), 400

    # TODO: Enforce user_id must equal current or seen uuid or fail request.
    user_id = user_id.strip()
    if user_id == "none":
        app_logger.warning("Chat request user_id cannot be none")
        return jsonify({
            "ok": False,
            "error": "Field 'user_id' cannot be none or empty"
        }), 400

    # rate limit check. blocks if user sending too frequently.
    if not rate_limiter.check(user_id):
        app_logger.warning("Chat request user_id was rate limited")
        return jsonify({
            "ok": False,
            "error": "Chat request was rate limited. Please wait 30 seconds before resubmission."
        }), 429
    else:
        app_logger.info("Rate limiting check passed")

    try:
        app_logger.info(f"Inserting new job for user_id {user_id} into db...")
        job_id = db.insert_job(user_id, msg)
        app_logger.info(f"Job write to db was successful, job_id {job_id}.")
    except Exception as e:
        app_logger.warning(f"Chat failed due to db job insert problem: {e}")

        return jsonify({
            "ok": False,
            "error": "Could not insert job to database",
            "job_id": "none",
            "user_id": user_id,
            "detail": ""
        }), 500

    model_ready = _async_to_sync(rag_controller.is_model_ready)()
    if not model_ready or force_queue:
        preview_msg = msg[:40]
        app_logger.warning(f"Model is not ready. Queueing msg for user {user_id}. Msg: {preview_msg}...  ")

        # send wake-up request to model
        _async_to_sync(rag_controller.send_warmup)()
        rag_controller.queue_job(user_id, job_id, msg)

        return jsonify({
            "ok": False,
            "error": "Model not ready. Job was queued.",
            "job_id": job_id,
            "user_id": user_id,
            "detail": ""
        }), 200
    else:
        app_logger.info("Model is ready for requests")

    try:
        app_logger.info("Triggering chat_with_corpus...")

        # track the inflight requests for queue depth reporting
        inflight_chat_reqs.inc()

        # LLM model call goes here
        answer, docs = chat_with_corpus(msg, top_k=10)

        app_logger.info(f"Marking job {job_id} as done in db...")
        db.mark_done(job_id, answer)

        inflight_chat_reqs.dec()
    except RuntimeError as e:
        app_logger.error("chat_with_corpus failed: {e}")

        inflight_chat_reqs.dec()
        return jsonify({
            "ok": False,
            "error": "Search system not initialized",
            "job_id": job_id,
            "user_id": user_id,
            "detail": str(e)
        }), 503
    except Exception as e:
        app_logger.error("chat_with_corpus failed: {e}")
        app_logger.warning(f"Chat failed with exception: {e}")

        inflight_chat_reqs.dec()
        return jsonify({
            "ok": False,
            "error": "Chat failed",
            "job_id": job_id,
            "user_id": user_id,
            "detail": str(e)
        }), 500

    app_logger.info("chat operation was completed successfully")
    return jsonify({
        "ok": True,
        "reply": answer,
        "job_id": job_id,
        "user_id": user_id,
        "detail": "success",
        "references": docs
    }), 200


# ------------------ Routes: A/B (gated, simple dev version) ------------------

@app.post("/api/ab")
def api_ab():
    locked = _require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    msg = (payload.get("message") or payload.get("query") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "Field 'message' must be a non-empty string"}), 400

    # Simple dev A/B: ask twice with slightly different prompts.
    # Replace later with true multi-model or different temperatures.
    try:
        a = ask_corpus(msg)
        b = ask_corpus(msg + "\n\n(Provide an alternate phrasing / approach.)")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": "Search system not initialized", "detail": str(e)}), 503
    except Exception as e:
        app_logger.exception("AB failed")
        return jsonify({"ok": False, "error": "AB failed", "detail": str(e)}), 500

    return jsonify({
        "ok": True,
        "a": a,
        "b": b,
        "references": [],
    }), 200


# ------------------ Routes: Feedback + Status + Queue ------------------

MODEL_CHECK_TIMEOUT_SECS = 5


@app.route("/api/feedback", methods=["POST", "GET"])
def api_feedback():
    return jsonify({"ok": True, "status": "feedback was successful"}), 200


@app.route("/api/status", methods=["GET", "POST"])
def api_status():
    model_ready = _async_to_sync(rag_controller.is_model_ready)()
    unlocked = _is_unlocked()
    app_logger.info(f"Status request was received. is_unlocked: {unlocked}, model_ready: {model_ready}")

    return jsonify({
        "ok": True,
        "status": "status was successful",
        "unlocked": _is_unlocked(),
        "retrieval_state_ready": retrieval_state is not None,
        "model_ready": model_ready,
    }), 200


# TODO: The below may not be accurate for queries that are purely in-flight and not stored in the queue.
@app.post("/api/queue")
def api_queue():
    """
    This allows clients to get info on queue depth and endpoint status.
    Logged-in clients are able to get the status of their requests and any pending responses.
    """
    global inflight_chat_reqs

    locked = _require_unlocked()
    model_ready = _async_to_sync(rag_controller.is_model_ready)()
    queue_len = rag_controller.job_queue_len() + inflight_chat_reqs.get()
    resp_len = rag_controller.outgoing_queue_len()

    payload = request.get_json(silent=True) or {}
    user_id = (payload.get("user_id") or "").strip()

    app_logger.info(f"locked: {not locked}, user_id: {user_id}")
    if not locked and user_id:
        # This is logged-in mode. It will fetch any user specific info and outgoing responses.
        app_logger.info(f"processing api_queue() request in logged-in mode for user {user_id}")

        job_index, queued_job = rag_controller.fetch_queued_job_info(user_id)
        queued_resp = rag_controller.fetch_queued_outgoing_info(user_id)

        app_logger.info(f"queued_job found: {queued_job is not None}, queued_resp found: {queued_resp is not None}")
        if queued_job is not None or queued_resp is not None:
            app_logger.info(f"Queued response was found for used_id {user_id}")

            resp_json = ""
            if queued_resp is not None:
                resp_json = queued_resp.to_json()

            return jsonify({
                "ok": True,
                "queries_in_line": queue_len,
                "resps_in_line": resp_len,
                "job_in_line": job_index,
                "outgoing_resp": resp_json,
                "model_ready": model_ready,
                "server_time": time.time()
            }), 200
        else:
            app_logger.info(f"No queued response was found for used_id {user_id}. Processing for general.")


    # This is the general view for those that aren't logged in.
    app_logger.info(f"processing api_queue() request in general mode")

    return jsonify({
        "ok": True,
        "queries_in_line": queue_len,
        "resps_in_line": resp_len,
        "model_ready": model_ready,
        "server_time": time.time()
    }), 200


# ------------------ Local dev runner ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
