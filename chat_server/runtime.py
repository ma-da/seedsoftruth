"""Process-wide runtime singletons and request-gating helpers.

Holds the objects that must be shared between the Flask app object
(:mod:`app`) and the API route handlers (:mod:`routes`) without either
importing the other: the application logger, the per-user rate limiter,
the inflight-request counter, the IP-unlock tracker, and the time-gate
helpers. Everything here is built once, at import time.
"""

import logging
import os
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

import auth
import config
import utils

# ------------------ Logging ------------------


def _setup_logger() -> logging.Logger:
    """Build the application logger.

    Prefers the project's ``logging_config.setup_logging`` if importable;
    otherwise falls back to a basic stderr handler at INFO level.

    Returns:
        The configured logger for the app.
    """
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


# ------------------ Shared singletons ------------------

# Inflight chat-request counter (thread-safe).
inflight_chat_reqs = utils.SafeInt(0)

# Per-user rate limiter.
rate_limiter = utils.SimpleUserRateLimiter(config.RATE_LIMITING_INTERVAL)


# Local-dev convenience: when running under flask debug mode, cap the IP
# lockout window so a developer iterating on the unlock flow is not
# locked out for the full production window. ``app.debug`` is itself
# derived from FLASK_DEBUG at startup, so reading the env var here is
# equivalent to the previous ``bool(app.debug)`` check and keeps this
# module independent of the Flask app object.
_flask_debug_env = (os.environ.get("FLASK_DEBUG", "") or "").strip().lower()
_debug_mode = _flask_debug_env in ("1", "true", "yes", "on")
if _debug_mode and config.IP_LOCKOUT_SECS > config.DEBUG_LOCKOUT_CAP_SECS:
    app_logger.info(
        "FLASK_DEBUG detected; capping IP lockout window at %ds "
        "(configured SOT_IP_LOCKOUT_SECS=%d) for local-dev ergonomics.",
        config.DEBUG_LOCKOUT_CAP_SECS,
        config.IP_LOCKOUT_SECS,
    )
    _EFFECTIVE_IP_LOCKOUT_SECS = config.DEBUG_LOCKOUT_CAP_SECS
else:
    _EFFECTIVE_IP_LOCKOUT_SECS = config.IP_LOCKOUT_SECS

ip_unlock_tracker = auth.IPUnlockTracker(
    config.MAX_UNLOCK_ATTEMPTS, _EFFECTIVE_IP_LOCKOUT_SECS
)


# ------------------ Time gating ------------------


def is_within_service_hours(
    now: datetime | None = None,
    tz: ZoneInfo = config.EASTERN_TZ,
) -> bool:
    """
    Returns True if current time is within allowed service hours.
    """
    if not config.ENABLE_TIME_GATING:
        return True

    if now is None:
        now = datetime.now(tz)
    else:
        now = now.astimezone(tz)

    current = now.time()
    return config.START_TIME <= current < config.END_TIME


def no_time_gate(fn: Callable) -> Callable:
    """Decorator marking a Flask handler as exempt from the time gate.

    Args:
        fn: The route handler to exempt.

    Returns:
        The same handler, tagged with a ``_no_time_gate`` attribute that
        ``time_gate`` checks before enforcing service hours.
    """
    fn._no_time_gate = True
    return fn
