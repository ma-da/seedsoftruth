"""Application configuration.

Environment-derived constants and the small parser helpers that produce
them. Every value here is computed once, at import time, from environment
variables and is immutable for the life of the process.

Extracted from ``app.py`` so configuration lives in one place,
independent of the Flask app object and request handling.
"""

import ipaddress
import os
from datetime import time as dt_time
from typing import List
from zoneinfo import ZoneInfo

import logging_config

config_logger = logging_config.get_logger("config")


# ------------------ Rate limiting ------------------

# Minimum number of seconds between two accepted requests from one user.
RATE_LIMITING_INTERVAL = 30


# ------------------ Server time gating ------------------

ENABLE_TIME_GATING = False
EASTERN_TZ = ZoneInfo("America/New_York")
START_TIME = dt_time(9, 0)  # 9:00 AM
END_TIME = dt_time(22, 0)  # 10:00 PM


# ------------------ Feature flags ------------------

USE_DOUBLE_PROMPT = True


# ------------------ Retrieval-state initialization ------------------

# Cooldown, in seconds, between lazy retrieval-state init retries.
INIT_RETRY_COOLDOWN_S = int(os.environ.get("SOT_INIT_RETRY_COOLDOWN_S", "10"))


# ------------------ Unlock gate: debug ergonomics ------------------

# Under flask debug mode the IP lockout window is capped at this many
# seconds so a developer iterating on the unlock flow is not locked out
# for the full production window.
DEBUG_LOCKOUT_CAP_SECS = 60


# ------------------ Password gating ------------------


def _parse_passwords(env_value: str) -> List[str]:
    """Parse a comma-separated env value into a list of unlock passwords.

    Args:
        env_value: Raw comma-separated string (e.g. from SOT_PASSWORDS).

    Returns:
        The non-empty, whitespace-stripped passwords. Empty if unset.
    """
    return [p.strip() for p in (env_value or "").split(",") if p.strip()]


ALLOWED_PASSWORDS = _parse_passwords(os.environ.get("SOT_PASSWORDS", ""))


def _parse_max_unlock_attempts(env_value: str, default: int = 3) -> int:
    """Read SOT_MAX_UNLOCK_ATTEMPTS as a positive int.

    Falls back to ``default`` on missing / unparseable values and clamps
    to a minimum of 1 so the gate can never be configured into a state
    where every request is auto-locked-out.

    Args:
        env_value: Raw env value to parse.
        default: Value to use when ``env_value`` is missing or invalid.

    Returns:
        The parsed attempt limit (>= 1).
    """
    raw = (env_value or "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        config_logger.warning(
            "SOT_MAX_UNLOCK_ATTEMPTS=%r is not an int; using default %d",
            raw,
            default,
        )
        return default
    return max(1, n)


# Max wrong-password attempts per web session before that session is
# locked out for the remainder of its lifetime (even with the correct
# password). Configurable via SOT_MAX_UNLOCK_ATTEMPTS. Default: 3.
MAX_UNLOCK_ATTEMPTS = _parse_max_unlock_attempts(
    os.environ.get("SOT_MAX_UNLOCK_ATTEMPTS", "")
)


def _parse_positive_int(env_value: str, default: int, name: str) -> int:
    """Read an env value as a positive int, clamped to a minimum of 1.

    Args:
        env_value: Raw env value to parse.
        default: Value to use when ``env_value`` is missing or unparseable.
        name: Env var name, used only for the warning log on bad input.

    Returns:
        The parsed int (>= 1), or ``default`` on missing/invalid input.
    """
    raw = (env_value or "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
    except ValueError:
        config_logger.warning(
            "%s=%r is not an int; using default %d", name, raw, default
        )
        return default
    return max(1, n)


# How long, in seconds, an IP stays locked out after burning its
# unlock-attempt budget. Default: 24h. Configurable via
# SOT_IP_LOCKOUT_SECS. The tracker lives in process memory, so a
# worker restart also drops the lockout.
IP_LOCKOUT_SECS = _parse_positive_int(
    os.environ.get("SOT_IP_LOCKOUT_SECS", ""),
    default=24 * 60 * 60,
    name="SOT_IP_LOCKOUT_SECS",
)


def _parse_trusted_proxies(env_value: str) -> List[ipaddress._BaseNetwork]:
    """Parse SOT_TRUSTED_PROXIES into a list of network objects.

    Accepts comma-separated entries; each entry can be a bare IP
    (treated as a /32 or /128) or CIDR. Invalid entries are logged and
    skipped, never raised, so a typo doesn't take the app down.

    Args:
        env_value: Raw comma-separated env value.

    Returns:
        The parsed network objects; empty when unset.
    """
    nets: List[ipaddress._BaseNetwork] = []
    for raw in (env_value or "").split(","):
        s = raw.strip()
        if not s:
            continue
        try:
            if "/" in s:
                nets.append(ipaddress.ip_network(s, strict=False))
            else:
                ip = ipaddress.ip_address(s)
                nets.append(ipaddress.ip_network(f"{s}/{ip.max_prefixlen}"))
        except ValueError:
            config_logger.warning(
                "SOT_TRUSTED_PROXIES: skipping invalid entry %r", s
            )
    return nets


# Allowlist of proxy IPs/CIDRs whose X-Forwarded-For / X-Real-IP
# headers we honor. When empty (the default), we never trust those
# headers — the client IP is whatever the socket peer is. This is the
# safe default for local dev: there's no nginx in front, so a curl can
# trivially set XFF and evade the per-IP gate, but with an empty
# allowlist we ignore the header. In prod set this to "127.0.0.1"
# (nginx → gunicorn over loopback) and only nginx's headers get
# trusted. CIDRs work too: "127.0.0.0/8" or "10.0.0.0/8".
TRUSTED_PROXIES = _parse_trusted_proxies(
    os.environ.get("SOT_TRUSTED_PROXIES", "")
)

# Migration helper: the previous version of this code used a blunt
# SOT_TRUST_PROXY_HEADERS flag that trusted XFF unconditionally. That
# was spoofable in local dev. If anyone still has the old var set, log
# a loud warning so they migrate to the allowlist.
_legacy_trust = (os.environ.get("SOT_TRUST_PROXY_HEADERS") or "").strip()
if _legacy_trust:
    config_logger.warning(
        "SOT_TRUST_PROXY_HEADERS is deprecated and has NO EFFECT. "
        "Use SOT_TRUSTED_PROXIES (comma-separated IPs/CIDRs) instead. "
        "Behind nginx on the same box, set SOT_TRUSTED_PROXIES=127.0.0.1."
    )


# ------------------ Session keys ------------------

# Keys used by the unlock gate inside the Flask session cookie.
SESS_UNLOCKED = "unlocked"
SESS_FAIL_COUNT = "unlock_failed_attempts"
SESS_LOCKED_OUT = "unlock_locked_out"
