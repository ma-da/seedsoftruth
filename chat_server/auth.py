"""Authentication and access-control helpers for the unlock gate.

Three groups of functionality, extracted from ``app.py``:

  * session helpers — read the per-session unlock state stored in the
    Flask session cookie (``is_unlocked``, ``require_unlocked``, ...);
  * ``IPUnlockTracker`` — an in-memory, per-worker record of failed
    ``/api/unlock`` attempts keyed by client IP, a second line of
    defense behind the per-session lockout;
  * client-IP resolution — ``client_ip`` and ``is_admin_local_request``,
    which decide the trustworthy client address for a request.

None of this depends on the Flask app object; the app wires it together.
"""

import ipaddress
import threading
import time as time_module
from typing import Any, Dict, Optional, Tuple

from flask import jsonify, session

import config

# ------------------ Session helpers ------------------


def is_unlocked() -> bool:
    """Return True if the current web session has passed the password gate."""
    return bool(session.get(config.SESS_UNLOCKED, False))


def is_unlock_locked_out() -> bool:
    """True if this web session has burned its unlock budget."""
    return bool(session.get(config.SESS_LOCKED_OUT, False))


def unlock_attempts_remaining() -> int:
    """Wrong-password attempts left for this session (>= 0)."""
    used = int(session.get(config.SESS_FAIL_COUNT, 0) or 0)
    return max(0, config.MAX_UNLOCK_ATTEMPTS - used)


def require_unlocked() -> Optional[Tuple[Any, int]]:
    """Guard helper for gated routes.

    Returns:
        ``None`` when the session is unlocked. Otherwise a
        ``(json_response, 403)`` tuple the caller should return directly.
    """
    if not is_unlocked():
        return (
            jsonify({"ok": False, "error": "locked", "message": "Not today"}),
            403,
        )
    return None


# ------------------ IP-level unlock tracker ------------------
#
# In-memory, per-worker tracker of failed /api/unlock attempts keyed by
# client IP. This is a second line of defense behind the per-session
# lockout: a session cookie can be dropped to reset session state, but
# the IP tracker survives until either (a) the configured lockout
# window elapses (default 24h) or (b) the worker process restarts.
#
# Multi-worker caveat: under gunicorn with N workers, state is not
# shared across processes. A determined attacker could see up to
# N * config.MAX_UNLOCK_ATTEMPTS attempts (per IP) before being locked
# out from every worker, because each worker only sees the requests it
# routed. Cross-worker state would require a shared store (Redis or a
# DB table). This is consistent with the project's existing in-process
# state pattern (queue, rate limiter); flagged here so it can be
# hardened later if needed.


class IPUnlockTracker:
    """In-memory, per-worker tracker of failed /api/unlock attempts by IP.

    Acts as a second line of defense behind the per-session lockout: a
    session cookie can be dropped to reset session state, but this IP
    tracker survives until either the configured lockout window elapses
    or the worker process restarts. State is not shared across gunicorn
    workers; see the module comment above for the multi-worker caveat.

    All public methods are thread-safe via an internal lock.
    """

    def __init__(self, max_attempts: int, lockout_secs: int):
        """Initialize an empty tracker.

        Args:
            max_attempts: Failed attempts allowed per IP before lockout.
            lockout_secs: Duration, in seconds, an IP stays locked out.
        """
        self._max = max_attempts
        self._lockout_secs = lockout_secs
        self._lock = threading.Lock()
        # ip -> {"fails": int, "locked_at": Optional[float], "last_seen": float}
        self._state: Dict[str, Dict[str, Any]] = {}

    def _entry_expired(self, entry: Dict[str, Any], now: float) -> bool:
        """Return True if a tracker entry has aged out and can be dropped.

        Args:
            entry: A per-IP state dict from ``self._state``.
            now: Current epoch time, as returned by ``time.time()``.

        Returns:
            True if the entry's lockout (or, if not locked, its last-seen
            time) is at least one full lockout window in the past.
        """
        locked_at = entry.get("locked_at")
        if locked_at is None:
            # Not locked: forget it if it hasn't been touched in a full
            # lockout window (keeps the dict from growing without bound
            # from one-off typo attempts).
            return (now - entry.get("last_seen", now)) >= self._lockout_secs
        return (now - locked_at) >= self._lockout_secs

    def status(self, ip: str) -> Tuple[bool, int, int]:
        """Report the current lockout status for an IP.

        Side effect: drops the entry if its lockout has fully expired.

        Args:
            ip: Client IP address to look up.

        Returns:
            A ``(is_locked, seconds_remaining, fails_so_far)`` tuple.
        """
        now = time_module.time()
        with self._lock:
            entry = self._state.get(ip)
            if entry is None:
                return False, 0, 0
            if self._entry_expired(entry, now):
                self._state.pop(ip, None)
                return False, 0, 0
            locked_at = entry.get("locked_at")
            if locked_at is None:
                return False, 0, int(entry.get("fails", 0))
            return (
                True,
                int(self._lockout_secs - (now - locked_at)),
                int(entry.get("fails", 0)),
            )

    def register_failure(self, ip: str) -> Tuple[bool, int, int]:
        """Record a failed unlock attempt for an IP.

        Already-locked IPs do NOT have their counters bumped — they
        just keep returning locked status until the window elapses.

        Args:
            ip: Client IP address that failed to unlock.

        Returns:
            A ``(now_locked, attempts_remaining,
            seconds_remaining_if_locked)`` tuple.
        """
        now = time_module.time()
        with self._lock:
            entry = self._state.get(ip)
            if entry is not None and self._entry_expired(entry, now):
                # Old lockout has fully aged out; start fresh.
                self._state.pop(ip, None)
                entry = None
            if entry is None:
                entry = {"fails": 0, "locked_at": None, "last_seen": now}
                self._state[ip] = entry

            entry["last_seen"] = now
            if entry["locked_at"] is not None:
                # Still inside the active lockout window. Do not
                # extend it; just report current status.
                return (
                    True,
                    0,
                    int(self._lockout_secs - (now - entry["locked_at"])),
                )

            entry["fails"] = int(entry.get("fails", 0)) + 1
            remaining = max(0, self._max - entry["fails"])
            if entry["fails"] >= self._max:
                entry["locked_at"] = now
                return True, 0, self._lockout_secs
            return False, remaining, 0

    def clear(self, ip: str) -> None:
        """Wipe state for a single IP (called after a successful unlock)."""
        with self._lock:
            self._state.pop(ip, None)

    def clear_all(self) -> int:
        """Wipe state for ALL IPs. Returns count cleared. Admin-only."""
        with self._lock:
            n = len(self._state)
            self._state.clear()
            return n

    def prune(self) -> int:
        """Drop fully-expired entries. Called opportunistically."""
        now = time_module.time()
        with self._lock:
            stale = [
                ip
                for ip, e in self._state.items()
                if self._entry_expired(e, now)
            ]
            for ip in stale:
                self._state.pop(ip, None)
            return len(stale)

    def snapshot_size(self) -> int:
        """Return the number of IPs currently held in the tracker."""
        with self._lock:
            return len(self._state)


# ------------------ Client-IP resolution ------------------


def _peer_is_trusted_proxy(peer: str) -> bool:
    """True if the socket peer is in the SOT_TRUSTED_PROXIES allowlist."""
    if not config.TRUSTED_PROXIES or not peer or peer == "unknown":
        return False
    try:
        addr = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(addr in net for net in config.TRUSTED_PROXIES)


def is_admin_local_request(req) -> bool:
    """
    True if this request looks like it came from a process running on
    the server itself (e.g., ``scripts/reset_lockouts.py`` invoked via
    SSH), as opposed to a user request forwarded by nginx.

    Heuristic: peer is a loopback address AND no proxy headers are
    present. This works because in any prod deploy that fronts gunicorn
    with nginx, nginx is configured to set ``X-Forwarded-For`` on every
    forwarded request (the standard
    ``proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;``).
    A loopback request without those headers therefore could not have
    come through nginx — it must be a direct connection to gunicorn,
    which on a standard deploy means someone with shell access on the
    box.

    This is the authentication model used by tools like the Docker
    daemon's local socket and Postgres's "peer" auth: filesystem /
    process access to the host implies trust.

    Caveat: if gunicorn is bound to a public interface AND a remote
    attacker can reach it directly bypassing nginx, that attacker can
    also forge a "looks local" request by simply not sending the
    proxy headers. Always bind gunicorn to ``127.0.0.1`` in prod.
    """
    peer = (req.remote_addr or "").strip()
    if peer not in ("127.0.0.1", "::1"):
        return False
    if (req.headers.get("X-Forwarded-For") or "").strip():
        return False
    if (req.headers.get("X-Real-IP") or "").strip():
        return False
    return True


def client_ip(req) -> str:
    """
    Return the real client IP.

    Header-trust rule: only honor X-Forwarded-For / X-Real-IP when the
    request's *socket peer* (``request.remote_addr``) is itself a
    trusted proxy from ``SOT_TRUSTED_PROXIES``. Otherwise use the peer
    address directly. This is what nginx, Apache, and Werkzeug's
    ProxyFix all recommend: never trust a header just because it's
    present, because anyone who can talk to your socket can send
    whatever headers they want.

    Local dev (no proxy, empty allowlist) -> always uses remote_addr
    (typically 127.0.0.1). Spoofed XFF from a curl on your laptop is
    silently ignored.

    Falls back to ``"unknown"`` only if remote_addr is also missing;
    failures from "unknown" still get counted in the IP tracker, so
    an attacker who manages to suppress remote_addr can still be
    locked out collectively.
    """
    peer = (req.remote_addr or "unknown").strip()
    if _peer_is_trusted_proxy(peer):
        xff = (req.headers.get("X-Forwarded-For") or "").strip()
        if xff:
            # XFF can be a comma-separated chain: "client, proxy1, proxy2"
            first = xff.split(",")[0].strip()
            if first:
                return first
        real_ip = (req.headers.get("X-Real-IP") or "").strip()
        if real_ip:
            return real_ip
    return peer
