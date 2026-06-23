"""API route handlers for the Seeds of Truth Flask app.

Every HTTP endpoint is defined here on a single Flask ``Blueprint``
(``bp``), which :mod:`app` registers onto the application. Shared
singletons and helpers — the logger, the rate limiter, the IP-unlock
tracker, and the ``no_time_gate`` decorator — come from :mod:`runtime`.
"""

import hmac
import inspect
import json
import os
import time as time_module
import traceback
import uuid
from datetime import datetime

from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    session,
    stream_with_context,
)

import auth
import config
import corpus
import db
import email_responses
import model_adapters
import model_prompts
import rag_controller
import state
import utils
from runtime import (
    app_logger,
    inflight_chat_reqs,
    ip_unlock_tracker,
    is_within_service_hours,
    no_time_gate,
    rate_limiter,
)

bp = Blueprint("api", __name__)


@bp.get("/")
def index():
    """GET / — render the single-page app shell.

    Returns:
        The rendered ``templates/index.html`` page.
    """
    return render_template("index.html")


# ------------------ Routes: Auth ------------------


@bp.post("/api/unlock")
@no_time_gate
def api_unlock():
    """POST /api/unlock — validate a password and unlock the session.

    Body: ``{"password": str}``. Checks the per-IP and per-session
    lockout layers first, then compares the password against
    ``SOT_PASSWORDS`` with a constant-time compare. A correct password
    sets the session gate, clears both failure counters, and mints a new
    ``user_id``; a wrong/empty one counts against both lockout budgets.

    Returns:
        JSON ``(body, status)``: 200 with ``user_id`` on success, 403 on
        a non-locking failure, 429 once an IP or session is locked out.
    """
    payload = request.get_json(silent=True) or {}
    pw = (payload.get("password") or "").strip()
    ip = auth.client_ip(request)

    # ---- Layer 1: IP-level lockout (survives cookie clearing) ----
    ip_locked, ip_secs_left, _ = ip_unlock_tracker.status(ip)
    if ip_locked:
        app_logger.info(
            "Unlock refused: IP %s is locked out (~%ds remaining).",
            ip,
            ip_secs_left,
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "locked_out",
                    "message": "Too many failed attempts from your network. Try again later.",
                    "attempts_remaining": 0,
                    "locked_out": True,
                    "lockout_seconds_remaining": ip_secs_left,
                    "scope": "ip",
                }
            ),
            429,
        )

    # ---- Layer 2: session-level lockout (sticks until cookie dropped) ----
    if auth.is_unlock_locked_out():
        app_logger.info(
            "Unlock refused: session is locked out (max attempts = %d).",
            config.MAX_UNLOCK_ATTEMPTS,
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "locked_out",
                    "message": "Too many failed attempts. Try again in a new session.",
                    "attempts_remaining": 0,
                    "locked_out": True,
                    "scope": "session",
                }
            ),
            429,
        )

    # Missing password or no configured passwords -> count as a failed
    # attempt so blank-body probing can't be used to bypass the budget.
    if not pw or not config.ALLOWED_PASSWORDS:
        session[config.SESS_UNLOCKED] = False
        return _register_failed_unlock(ip=ip, reason="empty_or_unconfigured")

    ok = any(hmac.compare_digest(pw, real) for real in config.ALLOWED_PASSWORDS)

    if not ok:
        session[config.SESS_UNLOCKED] = False
        return _register_failed_unlock(ip=ip, reason="bad_password")

    # Success path: clear both failure counters (session + IP) so a
    # legitimate user with a typo earlier in the session doesn't get
    # surprise-locked-out on a future re-auth. Each unlock creates a
    # new uuid.
    # TODO: Make this robust against an attacker who farms uuids.
    new_uuid = uuid.uuid4()
    session[config.SESS_UNLOCKED] = True
    session.pop(config.SESS_FAIL_COUNT, None)
    ip_unlock_tracker.clear(ip)
    app_logger.info("Password accepted. Unlock succeeded for IP %s.", ip)

    return (
        jsonify(
            {
                "ok": True,
                "message": "Access Granted",
                "user_id": str(new_uuid),
                "attempts_remaining": config.MAX_UNLOCK_ATTEMPTS,
                "locked_out": False,
            }
        ),
        200,
    )


def _register_failed_unlock(ip: str, reason: str):
    """
    Increment BOTH the per-session and per-IP failure counters and
    return the response. Whichever layer locks out first wins (IP is
    checked first by api_unlock(), so IP lockout is what the client
    will see going forward).
    """
    # --- session layer ---
    sess_used = int(session.get(config.SESS_FAIL_COUNT, 0) or 0) + 1
    session[config.SESS_FAIL_COUNT] = sess_used
    sess_remaining = max(0, config.MAX_UNLOCK_ATTEMPTS - sess_used)
    sess_locked_now = sess_used >= config.MAX_UNLOCK_ATTEMPTS
    if sess_locked_now:
        session[config.SESS_LOCKED_OUT] = True

    # --- IP layer ---
    ip_locked_now, ip_remaining, ip_secs_left = (
        ip_unlock_tracker.register_failure(ip)
    )

    # Lockout-scope reporting: prefer IP if it tripped, since that's the
    # stronger / longer-lived gate the client now has to wait out.
    if ip_locked_now:
        app_logger.info(
            "Unlock failed (%s) from IP %s -> IP locked out for %ds. "
            "(session counter %d/%d)",
            reason,
            ip,
            ip_secs_left,
            sess_used,
            config.MAX_UNLOCK_ATTEMPTS,
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "locked_out",
                    "message": "Too many failed attempts from your network. Try again later.",
                    "attempts_remaining": 0,
                    "locked_out": True,
                    "lockout_seconds_remaining": ip_secs_left,
                    "scope": "ip",
                }
            ),
            429,
        )

    if sess_locked_now:
        app_logger.info(
            "Unlock failed (%s) from IP %s -> session locked out. "
            "(IP %d remaining)",
            reason,
            ip,
            ip_remaining,
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "locked_out",
                    "message": "Too many failed attempts. Try again in a new session.",
                    "attempts_remaining": 0,
                    "locked_out": True,
                    "scope": "session",
                }
            ),
            429,
        )

    # Neither layer locked yet. Report the tighter of the two remaining
    # budgets so the client sees the most urgent number.
    shown_remaining = min(sess_remaining, ip_remaining)
    app_logger.info(
        "Unlock failed (%s) from IP %s. session=%d/%d, ip=%d remaining.",
        reason,
        ip,
        sess_used,
        config.MAX_UNLOCK_ATTEMPTS,
        ip_remaining,
    )
    return (
        jsonify(
            {
                "ok": False,
                "message": "Not today",
                "attempts_remaining": shown_remaining,
                "locked_out": False,
            }
        ),
        403,
    )


@bp.get("/api/access")
@no_time_gate
def api_access():
    """GET /api/access — report the caller's unlock/lockout state.

    Returns:
        JSON ``(body, 200)`` with ``unlocked``, ``locked_out``,
        ``attempts_remaining``, ``max_attempts``, the IP-lockout fields,
        and ``scope`` ("ip", "session", or null) indicating which layer,
        if any, is currently locking the caller out.
    """
    ip = auth.client_ip(request)
    ip_locked, ip_secs_left, ip_fails = ip_unlock_tracker.status(ip)
    session_locked = auth.is_unlock_locked_out()
    locked_out = bool(session_locked or ip_locked)
    if locked_out:
        attempts_remaining = 0
    else:
        attempts_remaining = min(
            auth.unlock_attempts_remaining(),
            max(0, config.MAX_UNLOCK_ATTEMPTS - ip_fails),
        )
    return (
        jsonify(
            {
                "ok": True,
                "unlocked": auth.is_unlocked(),
                "locked_out": locked_out,
                "attempts_remaining": attempts_remaining,
                "max_attempts": config.MAX_UNLOCK_ATTEMPTS,
                "ip_locked_out": ip_locked,
                "ip_lockout_seconds_remaining": (
                    ip_secs_left if ip_locked else 0
                ),
                "scope": (
                    "ip"
                    if ip_locked
                    else ("session" if session_locked else None)
                ),
            }
        ),
        200,
    )


# ------------------ Routes: Admin ------------------
#
# /api/admin/clear-lockouts has a two-tier auth model:
#
#   1. Direct local requests (loopback peer + no proxy headers) are
#      allowed without a token. Rationale: those can only originate
#      from a process running on the server itself, and anyone with
#      shell access on the box can already do anything they want.
#      This is the path scripts/reset_lockouts.py uses.
#
#   2. Remote requests (anything that went through nginx, which adds
#      X-Forwarded-For) require a valid SOT_ADMIN_TOKEN matched with
#      hmac.compare_digest. If SOT_ADMIN_TOKEN is unset, remote access
#      is disabled entirely — only local-server requests can clear.
#
# To trigger a remote clear:
#
#   export SOT_ADMIN_TOKEN="$(python -c 'import secrets;print(secrets.token_hex(32))')"
#   curl -X POST https://your-host/api/admin/clear-lockouts \
#        -H 'Content-Type: application/json' \
#        -d '{"token":"<value>","ip":"1.2.3.4"}'

ADMIN_TOKEN = (os.environ.get("SOT_ADMIN_TOKEN") or "").strip()

if ADMIN_TOKEN:
    app_logger.info(
        "Admin endpoint: remote access enabled via SOT_ADMIN_TOKEN."
    )
else:
    app_logger.info(
        "Admin endpoint: SOT_ADMIN_TOKEN unset; remote clears disabled, "
        "only direct local requests (e.g. scripts/reset_lockouts.py) can clear."
    )


@bp.post("/api/admin/clear-lockouts")
@no_time_gate
def api_admin_clear_lockouts():
    """POST /api/admin/clear-lockouts — clear IP and/or session lockouts.

    Body: ``{"token"?: str, "ip"?: str, "session"?: bool}``. Authorized
    either as a direct local-server request or via ``SOT_ADMIN_TOKEN``
    (see the module comment above for the two-tier auth model). ``ip``
    may be a single address or ``"*"`` to clear every tracked IP.

    Returns:
        JSON ``(body, status)``: 200 with a ``cleared`` summary on
        success, 400 when nothing was requested, 403 when unauthorized.
    """
    peer = auth.client_ip(request)
    payload = request.get_json(silent=True) or {}
    is_local = auth.is_admin_local_request(request)

    # ---- Auth ----
    authed_via = None  # for logging only
    if is_local:
        authed_via = "local"
    elif ADMIN_TOKEN:
        provided = (payload.get("token") or "").strip()
        if provided and hmac.compare_digest(provided, ADMIN_TOKEN):
            authed_via = "token"

    if authed_via is None:
        app_logger.warning(
            "Admin clear-lockouts: denied (peer=%s, is_local=%s, token_set=%s).",
            peer,
            is_local,
            bool(ADMIN_TOKEN),
        )
        return jsonify({"ok": False, "message": "Not today"}), 403

    # ---- Action ----
    target_ip = (payload.get("ip") or "").strip()
    clear_session = bool(payload.get("session", False))

    if not target_ip and not clear_session:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "nothing to clear: provide 'ip' and/or 'session'",
                }
            ),
            400,
        )

    cleared = {"ips": 0, "all_ips": False, "session": False}

    if target_ip == "*":
        cleared["ips"] = ip_unlock_tracker.clear_all()
        cleared["all_ips"] = True
    elif target_ip:
        ip_unlock_tracker.clear(target_ip)
        cleared["ips"] = 1

    if clear_session:
        session.pop(config.SESS_LOCKED_OUT, None)
        session.pop(config.SESS_FAIL_COUNT, None)
        session[config.SESS_UNLOCKED] = False
        cleared["session"] = True

    app_logger.info(
        "Admin clear-lockouts (auth=%s, peer=%s): target_ip=%r clear_session=%s -> %s",
        authed_via,
        peer,
        target_ip,
        clear_session,
        cleared,
    )
    return (
        jsonify({"ok": True, "cleared": cleared, "authed_via": authed_via}),
        200,
    )


# ------------------ Routes: Demo ping ------------------


@bp.post("/api/ping")
@no_time_gate
def api_ping():
    """POST /api/ping — connectivity smoke-test endpoint.

    Returns:
        JSON ``{ok, received, server_time, message}`` echoing the posted
        body back along with the server's UTC timestamp.
    """
    data = request.get_json(force=True)
    return jsonify(
        {
            "ok": True,
            "received": data,
            "server_time": datetime.utcnow().isoformat() + "Z",
            "message": "Flask received your message successfully.",
        }
    )


# ------------------ Routes: Search (always allowed) ------------------


@bp.post("/api/search")
def api_search():
    """POST /api/search — retrieval-only corpus search (always allowed).

    Body: ``{"query": str, "top_k"|"max_n"?: int, "shard_k"?: int,
    "subsets"?: list, "rag_algo_type"?: int}``. Runs ``corpus.search_corpus``
    and returns cleaned reference documents; this endpoint is not gated
    by the password unlock. Any v4 min-gate metadata is passed through.

    Returns:
        JSON ``(body, status)``: 200 with ``references``/``results`` and
        ``num_results``, 400 on invalid fields, 500 on internal error.
    """
    try:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()

        # Back-compat: existing UI sends max_n; new UI can send top_k
        try:
            top_k = int(payload.get("top_k", payload.get("max_n", 10)))
        except Exception:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Field 'top_k'/'max_n' must be an integer",
                    }
                ),
                400,
            )

        # Closest shards to search (centroid routing count)
        try:
            shard_k = int(payload.get("shard_k", 20))
        except Exception:
            return (
                jsonify(
                    {"ok": False, "error": "Field 'shard_k' must be an integer"}
                ),
                400,
            )

        if not query:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Field 'query' must be a non-empty string",
                    }
                ),
                400,
            )
        if top_k <= 0 or top_k > 200:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Field 'top_k' must be between 1 and 200",
                    }
                ),
                400,
            )
        if shard_k <= 0 or shard_k > 200:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Field 'shard_k' must be between 1 and 200",
                    }
                ),
                400,
            )
        subsets = payload.get("subsets", None)

        rag_algo_type = payload.get("rag_algo_type", None)
        if rag_algo_type is None:
            rag_algo_type = 5
        else:
            rag_algo_type = int(rag_algo_type)

        # Call corpus.search_corpus in a compatible way
        sig = inspect.signature(corpus.search_corpus)
        params = sig.parameters
        rag_algo_choice = rag_algo_type

        if "shard_k" in params:
            results = corpus.search_corpus(
                query,
                top_k=top_k,
                shard_k=shard_k,
                subsets=subsets,
                rag_algo_choice=rag_algo_choice,
            )
        elif "centroid_k" in params:
            results = corpus.search_corpus(
                query,
                top_k=top_k,
                centroid_k=shard_k,
                subsets=subsets,
                rag_algo_choice=rag_algo_choice,
            )  # in case you used centroid_k
        else:
            # old signature: corpus.search_corpus(query, top_k)
            results = corpus.search_corpus(
                query, top_k, rag_algo_choice=rag_algo_choice
            )

        # Normalize results BEFORE .get calls (prevents HTML 500)
        if not isinstance(results, dict):
            results = {"results": []}

        out_list = results.get("results", [])
        if not isinstance(out_list, list):
            out_list = []

        cleaned_list = rag_controller.clean_rag_references(out_list)
        app_logger.info(f"Original search_references api_search: {out_list}")

        num_results = results.get("num_results")
        if not isinstance(num_results, int):
            num_results = len(cleaned_list)

        app_logger.info(
            f"\n\n*****\n Cleaned search_references api_search: {cleaned_list} \n\n*****"
        )

        message = results.get("message")
        if not isinstance(message, str) or not message.strip():
            message = f"Found {num_results} result(s)."

        # v4 min-gate metadata (absent for v1/v2/v3 — these keys will be missing
        # from the corpus.search_corpus result for non-gated variants).
        gate_keys = (
            "gate_decision",
            "gate_reason",
            "top1_score",
            "n_canonical_entities",
            "n_non_location_entities",
            "fts_branch_used",
            "score_floor",
            "pre_gate_n_results",
        )
        gate_payload = {k: results[k] for k in gate_keys if k in results}

        response = {
            "ok": True,
            "query": query,
            "num_results": num_results,
            "message": message,
            "references": cleaned_list,
            "results": cleaned_list,
            "top_k": top_k,
            "shard_k": shard_k,
        }
        response.update(gate_payload)
        return jsonify(response), 200

    except Exception as e:
        # Always return JSON so the frontend sees the real error.
        tb = traceback.format_exc()
        try:
            app_logger.exception("Search failed")
        except Exception:
            pass

        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Search failed",
                    "detail": str(e),
                    "traceback": tb,
                }
            ),
            500,
        )


# ------------------ Routes: Chat (gated) ------------------


@bp.post("/api/chat")
def api_chat():
    """POST /api/chat — enqueue an async RAG chat turn.

    Async chat: validates the request, persists a job row, enqueues it for
    the background worker, and returns 202 immediately with a ``job_id``.
    The client then polls ``GET /api/job/<job_id>`` until status is
    ``done`` or ``failed``.

    No HTTP request to this endpoint blocks on the model, which is what
    eliminates Cloudflare 524 (origin_response_timeout) on slow LLM calls.

    Requires an unlocked session. Body: ``{"message"|"query": str,
    "user_id": str, "model_type": str, "use_rag"?: bool, "subsets"?:
    list, "rag_algo_type"?: int, "prompt_type"?: int (1-indexed)}``.
    ``use_rag`` and ``prompt_type`` flow through to the worker via
    ``QueuedJob`` fields. ``use_double_prompt`` is still a module-level
    constant (``config.USE_DOUBLE_PROMPT``) because no client surface
    exposes it per-request.

    Returns:
        JSON ``(body, status)``: 202 with ``job_id``/``queue_position``
        when enqueued, 400 on bad input, 429 when rate-limited, 403 when
        locked, 500 on internal failure.
    """
    locked = auth.require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    use_rag = bool(payload.get("use_rag", True))
    msg = (payload.get("message") or payload.get("query") or "").strip()
    if not msg:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Field 'message' must be a non-empty string",
                }
            ),
            400,
        )

    # `force_queue` is preserved as a no-op for one release so older clients
    # that still send it don't get a validation error. Every request is
    # queued now; the flag is meaningless.
    _ = (payload.get("force_queue") or "").strip()

    user_id = payload.get("user_id", "none")
    if not isinstance(user_id, str):
        app_logger.warning("Chat request user_id was invalid")
        return (
            jsonify({"ok": False, "error": "Field 'user_id' must be a string"}),
            400,
        )

    # TODO: Enforce user_id must equal current or seen uuid or fail request.
    user_id = user_id.strip()
    if user_id == "none" or not user_id:
        app_logger.warning("Chat request user_id cannot be none")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Field 'user_id' cannot be none or empty",
                }
            ),
            400,
        )

    model_type = payload.get("model_type", None)
    if model_type is None:
        app_logger.warning("Chat request model_type was missing")
        return (
            jsonify({"ok": False, "error": "Field 'model_type' was missing"}),
            400,
        )

    subsets = payload.get("subsets", None)

    if not model_adapters.is_valid_model_type(model_type):
        app_logger.warning(f"Chat request model_type was invalid: {model_type}")
        return (
            jsonify({"ok": False, "error": "Field 'model_type' was invalid"}),
            400,
        )

    rag_algo_type = payload.get("rag_algo_type", None)
    if rag_algo_type is None:
        rag_algo_type = 5
    else:
        rag_algo_type = int(rag_algo_type)

    # prompt_type uses 1-indexed values on the wire (1..N where N = number of
    # configured system prompts). We subtract 1 to get the internal 0-indexed
    # offset. Validation is preserved even though the worker consumes this
    # value asynchronously, so old/new clients get consistent error messages.
    n_prompts = len(model_prompts.MODEL_SYSTEM_PROMPTS)
    raw_prompt_type = payload.get("prompt_type", None)
    prompt_type = 0  # default: V1 (matches the legacy synchronous path)
    if raw_prompt_type is not None:
        try:
            sent = int(raw_prompt_type)
        except (TypeError, ValueError):
            app_logger.warning(
                f"Chat request prompt_type was not an integer: {raw_prompt_type!r}"
            )
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Field 'prompt_type' must be an integer in 1..{n_prompts}",
                    }
                ),
                400,
            )
        prompt_type = sent - 1
        if prompt_type < 0 or prompt_type >= n_prompts:
            app_logger.warning(
                f"Chat request prompt_type was out of range: sent={sent}, valid=1..{n_prompts}"
            )
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Field 'prompt_type' must be in 1..{n_prompts}",
                    }
                ),
                400,
            )

    # Per-user rate limit. Async / polling does NOT relax this — a flood of
    # cheap 202s is still a flood from the worker's perspective.
    if not rate_limiter.check(user_id):
        app_logger.warning("Chat request user_id was rate limited")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Chat request was rate limited. Please wait 30 seconds before resubmission.",
                }
            ),
            429,
        )

    app_logger.info("Rate limiting check passed")

    job_id = "none"
    try:
        app_logger.info(f"Inserting new job for user_id {user_id} into db...")
        job_id = db.insert_job(user_id, msg)
        app_logger.info(f"Job write to db was successful, job_id {job_id}.")
    except Exception as e:
        app_logger.exception("DB insert failed")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Could not insert job to database",
                    "job_id": job_id,
                    "user_id": user_id,
                    "detail": str(e),
                }
            ),
            500,
        )

    # Snapshot model readiness only to label the queue_reason in the
    # response — the client uses this to choose the "warming up" vs
    # "queue busy" copy. We do NOT branch on it; both paths queue.
    # MODEL_TIMEOUT_SECS=5s ceiling makes this safe under the 30s gunicorn
    # timeout. We intentionally do NOT call send_warmup_for here: its
    # internal HTTP timeout is 60s (see HFEndpointLLM.send_warmup in
    # model_adapters.py), which would let a single cold-start request
    # exceed gunicorn's timeout and leave the DB row orphaned. The worker
    # thread already calls send_warmup_for in its not-ready loop, so the
    # warmup happens just as fast in practice.
    try:
        model_ready = rag_controller.is_model_type_ready(model_type)
    except Exception:
        app_logger.exception(
            "is_model_type_ready failed (best-effort, swallowed)"
        )
        model_ready = False
    queue_reason = "queue_busy" if model_ready else "model_warming"

    # Enqueue. If queue_job raises (lock contention, OOM, etc.) we must
    # NOT leave the DB row sitting at status='queued' — it would be
    # invisible to the worker (not in the in-mem queue) and the client
    # would poll forever. Mark it failed eagerly so the polling client
    # gets a clean error and the user can retry.
    try:
        queue_position = rag_controller.queue_job(
            user_id, job_id, model_type, msg, subsets, rag_algo_type,
            use_rag=use_rag, prompt_type=prompt_type,
        )
    except Exception as e:
        app_logger.exception(
            f"queue_job failed for job_id {job_id}; marking failed"
        )
        try:
            db.mark_failed(job_id, f"Could not enqueue job: {e}")
        except Exception:
            app_logger.exception(
                "Follow-up mark_failed also raised; row is orphaned until sweep"
            )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Could not enqueue job",
                    "job_id": job_id,
                    "user_id": user_id,
                    "detail": str(e),
                }
            ),
            500,
        )

    app_logger.info(
        f"Chat job queued. user_id={user_id}, job_id={job_id}, "
        f"position={queue_position}, reason={queue_reason}, "
        f"model_type={model_type}, msg='{msg[:40]}...'"
    )

    return (
        jsonify(
            {
                "ok": True,
                "status": "queued",
                "job_id": job_id,
                "user_id": user_id,
                # Hint for the client's polling cadence. Sending it in the
                # response (rather than hardcoding on the client) lets us
                # tune the polling rate server-side without shipping JS. The
                # path itself is fixed: GET /api/job/<job_id>?user_id=...
                "poll_interval_ms": 1500,
                # Tells the client whether to offer the "email me when ready"
                # flow. Server is the source of truth — a stale client that
                # ignored this and POSTed to /api/email_response anyway would
                # be rejected at that endpoint.
                "email_offer": email_responses.is_enabled(),
                "queue_reason": queue_reason,
                "queue_position": queue_position,
            }
        ),
        202,
    )


# ------------------ Routes: Chat streaming (SSE) ------------------

# Headers that keep Server-Sent Events flowing unbuffered. ``X-Accel-Buffering:
# no`` disables nginx response buffering; ``no-transform`` stops intermediaries
# from gzipping/altering the stream (which would defeat token-by-token flush).
_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame: ``event: <name>`` + one
    ``data: <json>`` line, terminated by a blank line.

    Args:
        event: The SSE event name (``chunk``/``done``/``queued``/``error``).
        data: JSON-serializable payload.

    Returns:
        The encoded SSE frame string.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@bp.post("/api/chat/stream")
def api_chat_stream():
    """POST /api/chat/stream — stream a chat turn token-by-token over SSE.

    The streaming sibling of :func:`api_chat`. It runs the same validation
    and rate limiting, then:

    - If the selected adapter can stream (implements ``generate_stream``) and
      the model is ready, it generates synchronously and streams the reply as
      ``chunk`` events, finishing with a ``done`` event that carries the full
      reply plus reference docs. The job row is persisted (``mark_done``) so
      history and ``/api/job`` stay consistent with the queued path.
    - Otherwise (non-streaming adapter or model not ready) it enqueues the job
      exactly like :func:`api_chat` and emits a single ``queued`` event so the
      client falls back to polling ``GET /api/job/<id>``.

    Wire protocol (each frame is ``event: <name>\\n`` then ``data: <json>\\n\\n``):

        event: chunk   data: {"text": "..."}                      # 0..N
        event: done    data: {"ok": true, "reply", "references", "job_id", ...}
        event: queued  data: {"ok": false, "job_id", "queue_position", ...}
        event: error   data: {"ok": false, "error", "detail", "job_id"}

    Pre-stream validation failures still reply as ordinary JSON (400/403/429).
    """
    locked = auth.require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    use_rag = bool(payload.get("use_rag", True))

    msg = (payload.get("message") or payload.get("query") or "").strip()
    if not msg:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Field 'message' must be a non-empty string",
                }
            ),
            400,
        )

    user_id = payload.get("user_id", "none")
    if not isinstance(user_id, str):
        return (
            jsonify({"ok": False, "error": "Field 'user_id' must be a string"}),
            400,
        )
    user_id = user_id.strip()
    if user_id == "none" or not user_id:
        return (
            jsonify(
                {"ok": False, "error": "Field 'user_id' cannot be none or empty"}
            ),
            400,
        )

    model_type = payload.get("model_type", None)
    if model_type is None:
        return (
            jsonify({"ok": False, "error": "Field 'model_type' was missing"}),
            400,
        )
    if not model_adapters.is_valid_model_type(model_type):
        return (
            jsonify({"ok": False, "error": "Field 'model_type' was invalid"}),
            400,
        )

    subsets = payload.get("subsets", None)

    rag_algo_type = payload.get("rag_algo_type", None)
    rag_algo_type = 5 if rag_algo_type is None else int(rag_algo_type)

    # prompt_type: 1-indexed on the wire, 0-indexed internally (mirror api_chat).
    n_prompts = len(model_prompts.MODEL_SYSTEM_PROMPTS)
    raw_prompt_type = payload.get("prompt_type", None)
    prompt_type = 0
    if raw_prompt_type is not None:
        try:
            sent = int(raw_prompt_type)
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Field 'prompt_type' must be an integer in 1..{n_prompts}",
                    }
                ),
                400,
            )
        prompt_type = sent - 1
        if prompt_type < 0 or prompt_type >= n_prompts:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"Field 'prompt_type' must be in 1..{n_prompts}",
                    }
                ),
                400,
            )

    if not rate_limiter.check(user_id):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Chat request was rate limited. Please wait 30 seconds before resubmission.",
                }
            ),
            429,
        )

    # Persist the job row up front (status='queued') so both the streaming and
    # the queued fallback paths share one id that history/poll can resolve.
    try:
        job_id = db.insert_job(user_id, msg)
    except Exception as e:
        app_logger.exception("chat/stream: DB insert failed")
        return (
            jsonify(
                {"ok": False, "error": "Could not insert job to database", "detail": str(e)}
            ),
            500,
        )

    model_adaptor = rag_controller.get_model_type(model_type)
    streaming_capable = hasattr(model_adaptor, "generate_stream")
    try:
        model_ready = rag_controller.is_model_type_ready(model_type)
    except Exception:
        app_logger.exception("chat/stream: is_model_type_ready failed (swallowed)")
        model_ready = False

    # ---- Fallback: non-streaming adapter or model not ready -> queue ----
    if not streaming_capable or not model_ready:
        try:
            queue_position = rag_controller.queue_job(
                user_id, job_id, model_type, msg, subsets, rag_algo_type,
                use_rag=use_rag, prompt_type=prompt_type,
            )
        except Exception as e:
            app_logger.exception("chat/stream: queue_job failed; marking failed")
            try:
                db.mark_failed(job_id, f"Could not enqueue job: {e}")
            except Exception:
                app_logger.exception("chat/stream: follow-up mark_failed raised")

            def _gen_err():
                yield _sse("error", {
                    "ok": False, "error": "Could not enqueue job",
                    "detail": str(e), "job_id": job_id,
                })
            return Response(
                stream_with_context(_gen_err()),
                mimetype="text/event-stream", headers=_SSE_HEADERS,
            )

        queue_reason = "queue_busy" if model_ready else "model_warming"
        app_logger.info(
            f"chat/stream queued (capable={streaming_capable}, ready={model_ready}). "
            f"user_id={user_id}, job_id={job_id}, position={queue_position}, "
            f"model_type={model_type}"
        )

        def _gen_queued():
            yield _sse("queued", {
                "ok": False,
                "status": "queued",
                "job_id": job_id,
                "user_id": user_id,
                "poll_interval_ms": 1500,
                "email_offer": email_responses.is_enabled(),
                "queue_reason": queue_reason,
                "queue_position": queue_position,
                "detail": "Model not ready or non-streaming; job queued.",
            })
        return Response(
            stream_with_context(_gen_queued()),
            mimetype="text/event-stream", headers=_SSE_HEADERS,
        )

    # ---- Streaming path ----
    try:
        db.mark_processing(job_id)  # best-effort queued->processing
    except Exception:
        app_logger.exception("chat/stream: mark_processing failed (non-fatal)")

    app_logger.info(
        f"chat/stream generating. user_id={user_id}, job_id={job_id}, "
        f"model_type={model_type}, use_rag={use_rag}, msg='{msg[:40]}...'"
    )

    def _gen_stream():
        inflight_chat_reqs.inc()
        try:
            prep = corpus.prepare_chat_prompt(
                model_type, msg,
                use_rag=use_rag,
                use_double_prompt=config.USE_DOUBLE_PROMPT,
                subsets=subsets,
                rag_algo_choice=rag_algo_type,
                prompt_type=prompt_type,
            )
            adaptor = prep["model_adaptor"]

            pieces = []
            for piece in adaptor.generate_stream(
                prep["user_content"],
                system_prompt=prep["system_prompt"],
                temperature=model_adapters.DEFAULT_TEMPERATURE,
                max_new_tokens=model_adapters.DEFAULT_MAX_TOKENS,
            ):
                if piece:
                    pieces.append(piece)
                    yield _sse("chunk", {"text": piece})

            answer = "".join(pieces).strip()
            if prep.get("truncated"):
                answer = "(Question truncated)\n\n" + answer

            references = []
            if use_rag:
                try:
                    raw_docs = corpus.fetch_chat_references(
                        msg, answer, subsets=subsets, rag_algo_choice=rag_algo_type,
                    )
                    references = rag_controller.clean_rag_references(raw_docs)
                except Exception:
                    app_logger.exception(
                        "chat/stream: reference fetch failed (non-fatal)"
                    )
                    references = []

            # Persist using the same JSON blob shape the worker writes, so
            # /api/job/<id> and history render identically to the queued path.
            try:
                db.mark_done(job_id, json.dumps({
                    "reply": answer, "references": references,
                }))
            except Exception:
                app_logger.exception("chat/stream: mark_done failed")

            yield _sse("done", {
                "ok": True,
                "status": "done",
                "reply": answer,
                "references": references,
                "job_id": job_id,
                "user_id": user_id,
                "detail": "success",
            })
        except Exception as e:
            app_logger.exception("chat/stream: generation failed")
            try:
                db.mark_failed(job_id, f"Chat failed: {e}")
            except Exception:
                app_logger.exception("chat/stream: mark_failed raised")
            yield _sse("error", {
                "ok": False,
                "error": "Chat failed",
                "detail": str(e),
                "job_id": job_id,
            })
        finally:
            inflight_chat_reqs.dec()

    return Response(
        stream_with_context(_gen_stream()),
        mimetype="text/event-stream", headers=_SSE_HEADERS,
    )


# ------------------ Routes: Job status (polled by client) ------------------


@bp.get("/api/job/<job_id>")
@no_time_gate
def api_job_status(job_id: str):
    """GET /api/job/<job_id> — return the current state of one chat job.

    Polled by the client at ~1.5s intervals (see /api/chat
    ``poll_interval_ms``) until status is ``done`` or ``failed``.

    Response shapes:

    - done:       {ok:true,  status:"done",       reply, references, created_at, completed_at}
    - failed:     {ok:false, status:"failed",     error, detail,    created_at, completed_at}
    - queued:     {ok:true,  status:"queued",     queue_position,   created_at}
    - processing: {ok:true,  status:"processing", created_at}
    - 404:        {ok:false, error:"Not found"}     (unknown id OR wrong user_id)

    HTTP status is 200 for any "found" row regardless of terminal state;
    callers discriminate on the ``status`` field. 404 is reserved for not
    found, to keep ownership-leak surface area small (we don't disclose
    whether the id exists for a different user).

    Ownership check: requires ``?user_id=...`` and validates against the
    row that owns the job (same rule as ``db.get_job``).
    """
    locked = auth.require_unlocked()
    if locked:
        return locked

    user_id = (request.args.get("user_id") or "").strip()
    if not user_id or user_id == "none":
        return (
            jsonify(
                {"ok": False, "error": "Query param 'user_id' is required"}
            ),
            400,
        )

    try:
        row = db.get_job(job_id, user_id)
    except Exception:
        app_logger.exception("api_job_status: db.get_job failed")
        return jsonify({"ok": False, "error": "Lookup failed"}), 500

    if row is None:
        # Include the asked-for job_id so a screenshot of the error tells
        # an operator which id was probed. We do NOT distinguish between
        # "unknown id" and "wrong user_id" — both look identical to the
        # client to keep ownership-leak surface area small.
        return (
            jsonify({"ok": False, "error": "Not found", "job_id": job_id}),
            404,
        )

    status = row.get("status")
    created_at = row.get("created_at")
    completed_at = row.get("completed_at")
    raw_response = row.get("response")

    if status == "done":
        # Worker stores `{"reply", "references"}` as a JSON blob in the
        # `response` column. Older rows (created before this change rolled
        # out) may contain a plain string answer; fall back to that.
        reply = raw_response or ""
        references: list = []
        if isinstance(raw_response, str):
            stripped = raw_response.lstrip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(raw_response)
                    if isinstance(parsed, dict):
                        reply = parsed.get("reply", "") or ""
                        refs_val = parsed.get("references")
                        if isinstance(refs_val, list):
                            references = refs_val
                except (ValueError, TypeError) as e:
                    # Stored bytes look like JSON (start with '{') but
                    # don't parse. Log so an operator can spot which
                    # rows have corrupt blobs, then surface as-is to
                    # the user rather than hiding the partial data.
                    app_logger.warning(
                        "api_job_status: response column for job_id=%s "
                        "looks like JSON but failed to parse: %s",
                        job_id,
                        e,
                    )

        # Delivery-latency log: how long the row sat in the DB at
        # status='done' before a client polled it. Together with the
        # worker's LATENCY line this gives the full user-perceived
        # picture: queue_wait + processing (worker) + delivery (here).
        # Best-effort — never fail the response over a logging hiccup.
        try:
            if completed_at:
                completed_dt = datetime.fromisoformat(completed_at)
                delivery_secs = (
                    datetime.utcnow() - completed_dt
                ).total_seconds()
                app_logger.info(
                    "DELIVERY job_id=%s user_id=%s delay=%.3fs "
                    "(time from db mark_done to first poll that saw it)",
                    job_id,
                    user_id,
                    max(0.0, delivery_secs),
                )
        except Exception:
            pass

        return (
            jsonify(
                {
                    "ok": True,
                    "status": "done",
                    "job_id": job_id,
                    "reply": reply,
                    "references": references,
                    "created_at": created_at,
                    "completed_at": completed_at,
                }
            ),
            200,
        )

    if status == "failed":
        return (
            jsonify(
                {
                    "ok": False,
                    "status": "failed",
                    "job_id": job_id,
                    "error": "Chat failed",
                    "detail": raw_response or "",
                    "created_at": created_at,
                    "completed_at": completed_at,
                }
            ),
            200,
        )

    if status == "processing":
        return (
            jsonify(
                {
                    "ok": True,
                    "status": "processing",
                    "job_id": job_id,
                    "created_at": created_at,
                }
            ),
            200,
        )

    # Default branch: 'queued'. Compute the live queue position by scanning
    # the in-memory queue — the DB row only knows it's queued, not where in
    # line. This is cheap (queue is at most a handful of items in practice).
    queue_position = rag_controller.queue_position_for_job(job_id)
    return (
        jsonify(
            {
                "ok": True,
                "status": "queued",
                "job_id": job_id,
                "queue_position": queue_position,
                "created_at": created_at,
            }
        ),
        200,
    )


# ------------------ Routes: Email response (queued only) ------------------

# Per-session abuse limit. The /api/chat path is already rate-limited per
# user_id; this is just a belt-and-suspenders cap so a session can't
# attach 1000 different addresses to 1000 different jobs in a tight loop.
EMAIL_RESPONSES_PER_SESSION_LIMIT = 20


@bp.post("/api/email_response")
def api_email_response():
    """
    Attach an email address to a queued job so the chat response (or a
    "couldn't process this" notice) is sent there on completion.

    Body: {"job_id": str, "user_id": str, "email": str}

    Returns 200 with status="attached" if the job is still
    queued/processing, or status="sent_now" if the job already finished
    and we kicked off the email immediately.

    Disabled (404) when the email-responses feature flag is off.
    """
    if not email_responses.is_enabled():
        # 404 rather than 403 so a stale client treats this as "feature
        # doesn't exist here" and stops trying.
        return jsonify({"ok": False, "error": "feature_disabled"}), 404

    payload = request.get_json(silent=True) or {}
    job_id = (payload.get("job_id") or "").strip()
    user_id = (payload.get("user_id") or "").strip()
    raw_email = payload.get("email") or ""

    if not job_id:
        return (
            jsonify({"ok": False, "error": "Field 'job_id' is required"}),
            400,
        )
    if not user_id:
        return (
            jsonify({"ok": False, "error": "Field 'user_id' is required"}),
            400,
        )

    cleaned_email = email_responses.validate_email(raw_email)
    if not cleaned_email:
        return (
            jsonify(
                {"ok": False, "error": "Field 'email' is not a valid address"}
            ),
            400,
        )

    # Per-session belt-and-suspenders rate limit. Tracked in the Flask
    # session cookie itself — survives a refresh, doesn't pollute the DB.
    used = int(session.get("email_response_count", 0))
    if used >= EMAIL_RESPONSES_PER_SESSION_LIMIT:
        app_logger.warning(
            "email_response over per-session limit (used=%d, user_id=%s)",
            used,
            user_id,
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Too many email-response requests in this session. Try again later.",
                }
            ),
            429,
        )

    # Verify the job exists and belongs to this user_id. get_job filters
    # on (id, user_id) so a mismatch returns None.
    job = db.get_job(job_id, user_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404

    job_status = (job.get("status") or "").lower()

    # attach_email is now idempotent against 'sent'/'sending' — it
    # refuses to overwrite a row that's already been delivered, so a
    # duplicate POST short-circuits here without re-sending.
    attach_result = db.attach_email(job_id, cleaned_email)
    if attach_result == "not_found":
        return jsonify({"ok": False, "error": "Could not attach email"}), 500
    if attach_result == "already_sent":
        # Idempotent: a previous POST (or the worker) already delivered.
        # Return success so the client treats it as done — no need to
        # re-send and no need to confuse the user.
        return (
            jsonify(
                {
                    "ok": True,
                    "status": "already_sent",
                    "job_id": job_id,
                }
            ),
            200,
        )

    # Count this against the per-session limit only once we've actually
    # done a state change.
    session["email_response_count"] = used + 1

    # Race: job already finished between the 503 and this call. In that
    # case try to send immediately rather than waiting for a worker tick
    # that will never come. We must claim the row first to avoid racing
    # the worker's post-mark_done hook (which can fire concurrently).
    if job_status in ("done", "failed"):
        if not db.claim_email_for_send(job_id):
            # Worker already claimed it between our attach and claim —
            # let the worker deliver. Tell the client it's attached.
            return (
                jsonify(
                    {
                        "ok": True,
                        "status": "attached",
                        "job_id": job_id,
                    }
                ),
                200,
            )

        # We own the send. Pull the latest snapshot (response text)
        # from the row.
        snapshot = db.get_email_for_job(job_id) or {}
        prompt_text = snapshot.get("prompt") or job.get("prompt") or ""
        response_text = snapshot.get("response") or job.get("response") or ""
        failed = job_status == "failed"

        try:
            ok = email_responses.send_response_email(
                cleaned_email,
                prompt_text,
                response_text,
                references=None,  # not stored in DB; the in-tab UI has refs
                failed=failed,
            )
        except Exception:
            app_logger.exception(
                "send_response_email raised for job_id=%s (race-branch)",
                job_id,
            )
            db.mark_email_status(job_id, "failed")
            return (
                jsonify(
                    {
                        "ok": False,
                        "status": "send_failed",
                        "job_id": job_id,
                    }
                ),
                200,
            )

        db.mark_email_status(job_id, "sent" if ok else "failed")

        return (
            jsonify(
                {
                    "ok": True,
                    "status": "sent_now" if ok else "send_failed",
                    "job_id": job_id,
                }
            ),
            200,
        )

    # Normal case: job still queued or processing — email is attached;
    # worker's mark_done/mark_failed hook will pick it up.
    app_logger.info(
        "Attached email to queued job_id=%s (status=%s)",
        job_id,
        job_status,
    )

    return (
        jsonify(
            {
                "ok": True,
                "status": "attached",
                "job_id": job_id,
            }
        ),
        200,
    )


@bp.post("/api/email_response/cancel")
def api_email_response_cancel():
    """
    Lets the client withdraw an email request before the job completes.
    No-ops cleanly if the job already finished or was never attached.
    """
    if not email_responses.is_enabled():
        return jsonify({"ok": False, "error": "feature_disabled"}), 404

    payload = request.get_json(silent=True) or {}
    job_id = (payload.get("job_id") or "").strip()
    user_id = (payload.get("user_id") or "").strip()

    if not job_id or not user_id:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Fields 'job_id' and 'user_id' are required",
                }
            ),
            400,
        )

    job = db.get_job(job_id, user_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404

    db.clear_email(job_id)
    return jsonify({"ok": True, "status": "cancelled", "job_id": job_id}), 200


# ------------------ Routes: A/B (gated, simple dev version) ------------------


@bp.post("/api/ab")
def api_ab():
    """POST /api/ab — gated A/B comparison of two chat answers (dev).

    Requires an unlocked session. Body: ``{"message"|"query": str,
    "user_id": str, "use_rag"?: bool, "shard_k"?: int, "top_k"?: int,
    "rag_algo_type"?: int}``. Calls ``corpus.chat_with_corpus`` twice — once on
    the raw message and once on an alternate-phrasing prompt. Performs
    no queueing: if the model is not ready it returns an error instead.

    Returns:
        JSON ``(body, status)``: 200 with ``a``/``b`` answers and their
        reference lists, 503 when the model is cold, 400 on bad input,
        429 when rate-limited, 403 when locked, 500 on internal failure.
    """

    locked = auth.require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    use_rag = payload.get("use_rag", True)
    use_rag = bool(use_rag)
    msg = (payload.get("message") or payload.get("query") or "").strip()
    if not msg:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Field 'message' must be a non-empty string",
                }
            ),
            400,
        )

    user_id = payload.get("user_id", "none")
    if not isinstance(user_id, str):
        app_logger.warn("Chat request user_id was invalid")
        return (
            jsonify({"ok": False, "error": "Field 'user_id' must be a string"}),
            400,
        )

    # TODO: Enforce user_id must equal current or seen uuid or fail request.
    user_id = user_id.strip()
    if user_id == "none":
        app_logger.warning("Chat request user_id cannot be none")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "AB test failed. Field 'user_id' cannot be none or empty",
                }
            ),
            400,
        )

    # model must be ready for AB test
    app_logger.info(f"Health request received: user_id '{user_id}'")
    model_ready = rag_controller.is_model_ready()
    if not model_ready:
        preview_msg = msg[:40]
        app_logger.warning(
            f"Model is not ready. AB test unavailable for message '{preview_msg}...'"
        )

        # send wake-up request to model
        rag_controller.send_warmup()

        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Model not ready. AB test unavailable.",
                    "job_id": "none",
                    "user_id": user_id,
                    "detail": "",
                }
            ),
            503,
        )
    else:
        app_logger.info("Model is ready for requests")

    # rate limit check. blocks if user sending too frequently.
    if not rate_limiter.check(user_id):
        app_logger.warning("Chat request user_id was rate limited")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "AB Chat request was rate limited. Please wait 30 seconds before resubmission.",
                }
            ),
            429,
        )
    else:
        app_logger.info("Rate limiting check passed")

    inflight_chat_reqs.inc()
    inflight_chat_reqs.inc()

    # Get two job ids for use, fail if either one doesn't work
    job_id_a = "none"
    job_id_b = "none"
    try:
        app_logger.info(
            f"Inserting jobs for user_id {user_id} for AB testing into db..."
        )
        job_id_a = db.insert_job(user_id, msg)
        job_id_b = db.insert_job(user_id, msg)
        app_logger.info(
            f"Job write to db for AB testing was successful, job_id_a {job_id_a}, job_id_b {job_id_b}."
        )
    except Exception as e:
        app_logger.warning(f"AB test failed due to db job insert problem: {e}")

        inflight_chat_reqs.dec()
        inflight_chat_reqs.dec()

        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Could not insert job to database",
                    "user_id": user_id,
                    "detail": "",
                }
            ),
            500,
        )

    # Simple dev A/B: ask twice with slightly different prompts.
    # Replace later with true multi-model or different temperatures.
    # Let client override shard_k/top_k for testing
    try:
        shard_k = int(payload.get("shard_k", 20))
        top_k = int(payload.get("top_k", 10))
    except Exception:
        shard_k, top_k = 20, 10

    # Forward the same rag_algo_type the client sent
    # (defaults to 5 = production default — V5 topic-boosted V4 + min-gate).
    rag_algo_type = payload.get("rag_algo_type", None)
    if rag_algo_type is None:
        rag_algo_type = 5
    else:
        try:
            rag_algo_type = int(rag_algo_type)
        except (TypeError, ValueError):
            rag_algo_type = 5
    rag_algo_choice = rag_algo_type

    # TODO: FIX ME LATER
    model_type_a = "hf"
    model_type_b = "hf"

    try:
        # A
        a_answer, a_refs = corpus.chat_with_corpus(
            model_type_a,
            msg,
            top_k=top_k,
            shard_k=shard_k,
            use_rag=use_rag,
            use_double_prompt=config.USE_DOUBLE_PROMPT,
            rag_algo_choice=rag_algo_choice,
        )

        # B (alternate phrasing prompt)
        b_prompt = msg + "\n\n(Provide an alternate phrasing / approach.)"
        b_answer, b_refs = corpus.chat_with_corpus(
            model_type_b,
            b_prompt,
            top_k=top_k,
            shard_k=shard_k,
            use_rag=use_rag,
            use_double_prompt=config.USE_DOUBLE_PROMPT,
            rag_algo_choice=rag_algo_choice,
        )

        app_logger.info(f"Marking job {job_id_a} as done in db...")
        db.mark_done(job_id_a, a_answer)
        app_logger.info(f"Marking job {job_id_b} as done in db...")
        db.mark_done(job_id_b, b_answer)

        app_logger.info(f"AB test done")

    except RuntimeError as e:
        app_logger.exception("AB failed")
        inflight_chat_reqs.dec()
        inflight_chat_reqs.dec()

        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Search system not initialized",
                    "detail": str(e),
                }
            ),
            503,
        )
    except Exception as e:
        inflight_chat_reqs.dec()
        inflight_chat_reqs.dec()

        app_logger.exception("AB failed")
        return (
            jsonify({"ok": False, "error": "AB failed", "detail": str(e)}),
            500,
        )

    inflight_chat_reqs.dec()
    inflight_chat_reqs.dec()

    return (
        jsonify(
            {
                "ok": True,
                "a": a_answer,
                "b": b_answer,
                "references_a": a_refs,
                "references_b": b_refs,
                "job_id_a": job_id_a,
                "job_id_b": job_id_b,
            }
        ),
        200,
    )


# ------------------ Routes: Feedback + Status + Queue ------------------

MODEL_CHECK_TIMEOUT_SECS = 5


@bp.route("/api/feedback", methods=["POST", "GET"])
def api_feedback():
    """POST|GET /api/feedback — record user ratings for a completed job.

    Requires an unlocked session. Body: ``{"job_id": str, "relevance":
    int, "accuracy": int, "style": int, "comments"?: str}``. Validates
    that the job exists, then writes the feedback row to the DB.

    Returns:
        JSON ``(body, status)``: 200 on success, 400 on missing/invalid
        fields or an unknown ``job_id``, 403 when locked, 500 on a DB
        write error.
    """

    locked = auth.require_unlocked()
    if locked:
        return locked

    payload = request.get_json(silent=True) or {}
    job_id = utils.get_payload_str(payload, "job_id", None)
    if not job_id or len(job_id) == 0:
        app_logger.warning("Feedback request failed due to bad job_id param")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Field 'job_id' must be a non-empty string",
                }
            ),
            400,
        )

    relevance = utils.get_payload_int(payload, "relevance", None)
    if not relevance:
        app_logger.warning("Feedback request failed due to bad relevance param")
        return (
            jsonify(
                {"ok": False, "error": "Field 'relevance' must be integer"}
            ),
            400,
        )

    accuracy = utils.get_payload_int(payload, "accuracy", None)
    if not accuracy:
        app_logger.warning("Feedback request failed due to bad accuracy param")
        return (
            jsonify({"ok": False, "error": "Field 'accuracy' must be integer"}),
            400,
        )

    style = utils.get_payload_int(payload, "style", None)
    if not style:
        app_logger.warning("Feedback request failed due to bad style param")
        return (
            jsonify({"ok": False, "error": "Field 'style' must be integer"}),
            400,
        )

    comments = utils.get_payload_str(payload, "comments", "")
    app_logger.info(
        f"feedback request received for job_id {job_id}, comments {comments[:20]}"
    )

    # job id should exist in jobs table
    if not db.job_exists(job_id):
        app_logger.warning(
            f"Feedback request failed due to non-existent job_id: {job_id}"
        )
        return jsonify({"ok": False, "error": "job_id was invalid"}), 400

    try:
        app_logger.info(
            f"Feedback request received: job_id '{job_id}', relevance {relevance}, accuracy {accuracy}, style {style}, comments '{comments}' "
        )
        db.insert_feedback(job_id, relevance, accuracy, style, comments)
        app_logger.info("Feedback was successfully saved")
    except Exception as e:
        app_logger.error(f"insert feedback failed: {e}")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "feedback write failed",
                    "detail": str(e),
                }
            ),
            500,
        )

    return jsonify({"ok": True, "status": "feedback was successful"}), 200


@bp.route("/api/status", methods=["GET", "POST"])
@no_time_gate
def api_status():
    """GET|POST /api/status — report app and model health.

    Body (optional): ``{"health"?: str, "user_id"?: str, "model_type"?:
    str}``. When ``health`` is truthy and the request is within service
    hours, probes the selected (or default) model backend; otherwise the
    model is reported as ``"unchecked"``.

    Returns:
        JSON ``(body, 200)`` with ``unlocked`` and
        ``retrieval_state_ready``, plus ``model_ready`` when a health
        check was actually performed.
    """
    unlocked = auth.is_unlocked()

    payload = request.get_json(silent=True) or {}
    health_str = (payload.get("health") or "").strip()
    health = utils.str_to_bool(health_str, strict=False)
    user_id = (payload.get("user_id") or "").strip()
    model_type = (payload.get("model_type") or "").strip()

    is_within_hours = is_within_service_hours()
    if health and is_within_hours:
        app_logger.info("status health checked")
        if model_type and model_adapters.is_valid_model_type(model_type):
            model_ready = rag_controller.is_model_type_ready(model_type)
        else:
            model_ready = rag_controller.is_model_ready()
    else:
        app_logger.info("status health unchecked")
        model_ready = "unchecked"

    app_logger.info(
        f"status request received for user_id: {user_id}, model_type: {model_type or '(default)'}, is_unlocked: {unlocked}, model_ready: {model_ready}, health {health}, is_within_hours {is_within_hours}"
    )

    if model_ready == "unchecked":
        app_logger.info("api_status model_ready unchecked")
        return (
            jsonify(
                {
                    "ok": True,
                    "status": "status was successful",
                    "unlocked": auth.is_unlocked(),
                    "retrieval_state_ready": state.get_retrieval_state()
                    is not None,
                }
            ),
            200,
        )
    else:
        app_logger.info(f"api_status model_ready {model_ready}")
        return (
            jsonify(
                {
                    "ok": True,
                    "status": "status was successful",
                    "unlocked": auth.is_unlocked(),
                    "retrieval_state_ready": state.get_retrieval_state()
                    is not None,
                    "model_ready": model_ready,
                }
            ),
            200,
        )


@bp.post("/api/queue")
@no_time_gate
def api_queue():
    """POST /api/queue — queue-depth widget.

    Returns aggregate counters only — ``queries_in_line`` is in-mem queue
    length plus in-flight work in the worker.

    As of the async-chat migration, this endpoint NO LONGER carries the
    user's outgoing response. Per-job state belongs to ``/api/job/<id>``;
    keeping the answer here too would create a second source of truth
    that can drift from the DB.

    ``user_id`` is accepted for backwards compatibility (older clients send
    it) but is now unused.
    """
    _ = auth.require_unlocked()  # not enforced; this is a public widget endpoint
    queue_len = rag_controller.job_queue_len() + inflight_chat_reqs.get()

    return (
        jsonify(
            {
                "ok": True,
                "queries_in_line": queue_len,
                # Kept for client compatibility — there's no separate outgoing
                # queue anymore, so this is always 0.
                "resps_in_line": 0,
                "email_offer": email_responses.is_enabled(),
                "server_time": time_module.time(),
            }
        ),
        200,
    )
