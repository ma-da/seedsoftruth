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
import json
import random
import threading
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from flask import Flask, render_template, request, jsonify, session

import traceback
import inspect

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
            import rag_controller  # type: ignore
        except Exception as e:
            _last_init_error = f"rag_controller import failed: {e}"
            app_logger.exception(_last_init_error)
            retrieval_state = None
            return False

        try:
            boot = getattr(rag_controller, "boot", None)
            if boot is None:
                _last_init_error = "rag_controller.boot not found"
                app_logger.error(_last_init_error)
                retrieval_state = None
                return False

            retrieval_state = boot()
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

def ensure_state() -> bool:
    """Lazy initializer used by routes. Returns True if ready."""
    return init_state(force=False)

# Eager init once at import time (fine under systemd + gunicorn).
# If this fails (network down, etc.), ensure_state() will keep retrying on requests.
init_state(force=False)
db.init_db()

# ------------------ Async bridge helpers ------------------

def _async_to_sync(coro_fn):
    """
    Convert an async function (no args) to sync call.
    Prefers asgiref, which is the most reliable for sync Flask under gunicorn.
    """
    try:
        from asgiref.sync import async_to_sync  # type: ignore
        return async_to_sync(coro_fn)
    except Exception:
        # fallback: run in a fresh event loop (ok for dev; not ideal at scale)
        import asyncio
        def _runner():
            return asyncio.run(coro_fn())
        return _runner

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

    import rag_controller  # must be importable in this environment

    from asgiref.sync import async_to_sync  # pip install asgiref

    q = (query or "").strip()
    if not q:
        return ("", [])

    async def _run():
        # ---------- 1) Get answer ----------
        # Prefer ask() if present (your current controller showed this exists)
        if hasattr(rag_controller, "ask"):
            answer = await rag_controller.ask(retrieval_state, q, verbose=False)
        else:
            # Fallback path if your controller is more "primitive"
            if not hasattr(rag_controller, "sparse_retrieve") or not hasattr(rag_controller, "build_context"):
                raise AttributeError("rag_controller must expose ask() OR (sparse_retrieve + build_context + LLM call)")

            docs_for_answer = await rag_controller.sparse_retrieve(retrieval_state, q)
            context = rag_controller.build_context(docs_for_answer, q)

            # Try common LLM-call function names if ask() isn't present
            if hasattr(rag_controller, "ask_hf"):
                answer = await rag_controller.ask_hf(q, context)
            elif hasattr(rag_controller, "ask_llm"):
                # only used if it exists; fixes your current AttributeError
                answer = await rag_controller.ask_llm(q, context)
            else:
                raise AttributeError("rag_controller has no ask()/ask_hf()/ask_llm() to generate an answer")

        # ---------- 2) Get docs for references ----------
        # Prefer a dedicated search_references() if present, since it often returns richer metadata
        docs = []
        if hasattr(rag_controller, "search_references"):
            pack = await rag_controller.search_references(retrieval_state, q, top_k=top_k)
            if isinstance(pack, dict):
                docs = pack.get("results", []) or []
            elif isinstance(pack, list):
                docs = pack
        elif hasattr(rag_controller, "sparse_retrieve"):
            docs = await rag_controller.sparse_retrieve(retrieval_state, q)
        else:
            docs = []

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

    ok = any(hmac.compare_digest(pw, real) for real in ALLOWED_PASSWORDS)
    session["unlocked"] = bool(ok)

    return (jsonify({"ok": True, "message": "Access Granted"}), 200) if ok else \
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
def on_search():
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
def on_chat():
    locked = _require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    msg = (payload.get("message") or payload.get("query") or "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "Field 'message' must be a non-empty string"}), 400

    try:
        answer, docs = chat_with_corpus(msg, top_k=10)

    except RuntimeError as e:
        return jsonify({
            "ok": False,
            "error": "Search system not initialized",
            "detail": str(e)
        }), 503
    except Exception as e:
        app_logger.exception("Chat failed")
        return jsonify({
            "ok": False,
            "error": "Chat failed",
            "detail": str(e)
        }), 500

    return jsonify({
        "ok": True,
        "reply": answer,
        "references": docs,   # ✅ THIS IS WHAT WAS MISSING
    }), 200

# ------------------ Routes: A/B (gated, simple dev version) ------------------

@app.post("/api/ab")
def on_ab():
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

@app.route("/api/feedback", methods=["POST", "GET"])
def on_feedback():
    return jsonify({"ok": True, "status": "feedback was successful"}), 200

@app.route("/api/status", methods=["GET", "POST"])
def on_status():
    return jsonify({
        "ok": True,
        "unlocked": _is_unlocked(),
        "retrieval_state_ready": retrieval_state is not None,
        "last_init_error": _last_init_error,
    }), 200

@app.get("/api/queue")
def api_queue():
    # Dev stub — shows UI movement
    return jsonify({
        "ok": True,
        "queries_in_line": random.randint(0, 7),
        "server_time": time.time()
    }), 200

# ------------------ Local dev runner ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
