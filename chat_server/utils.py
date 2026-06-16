"""Shared, dependency-light helper utilities used across the app.

Includes boolean/number parsing for request payloads, a token-count
estimate, a thread-safe integer wrapper, a simple per-user rate limiter,
and an async-to-sync bridge for calling coroutines from sync Flask code.
"""

import threading
import time

from asgiref.sync import async_to_sync  # pip install asgiref

import logging_config

utils_logger = logging_config.get_logger("utils")


def str_to_bool(s: str, strict: bool = True) -> bool:
    """Parses a string into a boolean.

    Args:
        s: The string to parse, compared case-insensitively after
            surrounding whitespace is stripped.
        strict: When True, only ``"true"`` and ``"false"`` are accepted
            and any other value raises. When False, any value other
            than ``"true"`` returns False.

    Returns:
        The parsed boolean value.

    Raises:
        TypeError: If ``s`` is not a string.
        ValueError: If ``strict`` is True and ``s`` is neither
            ``"true"`` nor ``"false"``.
    """
    if not isinstance(s, str):
        raise TypeError("Expected string")

    val = s.strip().lower()
    if val == "true":
        return True

    if strict == True:
        if val == "false":
            return False

        raise ValueError(f"Invalid boolean string: {s!r}")
    else:
        return False


def estimate_tokens(text: str) -> int:
    """Roughly estimates the token count of a piece of text.

    Uses a words-times-1.3 heuristic as a cheap approximation of real
    tokenizer output; it is intentionally not exact.

    Args:
        text: The text to estimate.

    Returns:
        The estimated token count.
    """
    return int(len(text.split()) * 1.3)


class SafeInt:
    """
    A thread-safe int wrapper.
    Note the atomic.INT in the atomics package doesn't work for 3.13+ currently. This is a replacement.

    The lock is created per-instance in __init__; a class-level lock would
    be shared across every SafeInt and serialize unrelated counters.
    """

    def __init__(self, val):
        """Initialize the counter with ``val`` and a per-instance lock.

        Args:
            val: The initial integer value.
        """
        self.val = val
        self._lock = threading.Lock()

    def get(self):
        """Return the current value, holding the lock for the read."""
        with self._lock:
            return self.val

    def set(self, val):
        """Atomically set the value to ``val``."""
        with self._lock:
            self.val = val

    def inc(self):
        """Atomically increment the value by one."""
        with self._lock:
            self.val = self.val + 1

    def dec(self):
        """Atomically decrement the value by one."""
        with self._lock:
            self.val = self.val - 1


class SimpleUserRateLimiter:
    """
    A class that tracks rate limiting per user. Only allows one request per X seconds.

    Thread-safe: check() is wrapped in a per-instance lock so two
    concurrent requests for the same user_id can't both pass the
    check-then-update window. Required under gthread-style gunicorn
    workers where multiple request threads share one rate_limiter
    instance.
    """

    def __init__(self, interval_secs_param: int) -> None:
        """Initializes the rate limiter.

        Args:
            interval_secs_param: Minimum number of seconds that must
                elapse between two allowed requests from the same user.

        Raises:
            RuntimeError: If ``interval_secs_param`` is not a positive
                integer.
        """
        if not isinstance(interval_secs_param, int):
            raise RuntimeError("interval_secs_param must be int")
        if interval_secs_param <= 0:
            raise RuntimeError("interval_secs_param must be greater than zero")
        self.interval_secs = interval_secs_param
        # Per-instance state. The previous version declared user_data
        # at class scope, which would have caused every instance to
        # share the same dict if more than one were ever constructed.
        self.user_data = {}
        self._lock = threading.Lock()

    def check(self, user_id) -> bool:
        """
        Check if current user_id operation is allowed. Tracks current time versus last time.
        Returns true if time difference greater than interval, false if less
        """
        if not isinstance(user_id, str):
            raise RuntimeError("user_id must be str")
        if len(user_id) == 0:
            raise RuntimeError("user_id must not be empty")

        curr_time = int(time.time())

        with self._lock:
            last_time = self.user_data.get(user_id)

            if last_time is None:
                self.user_data[user_id] = curr_time
                return True

            assert curr_time >= last_time  # if not true, then its nonsensical
            diff = curr_time - last_time
            is_allowed = diff >= self.interval_secs
            if is_allowed:
                self.user_data[user_id] = curr_time
            return is_allowed


def get_payload_str(payload, key, default=""):
    """
    Helper for reading a string field from a requests payload
    """
    val = payload.get(key)
    return val.strip() if isinstance(val, str) else default


def get_payload_int(payload, key, default=0):
    """
    Helper for reading an int field from a requests payload
    """
    val = payload.get(key)
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return default
    return default


def get_payload_float(payload, key, default=0.0):
    """
    Helper for reading a float field from a requests payload
    """
    val = payload.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return default
    return default


def get_payload_bool(payload, key, default=False):
    """
    Helper for reading a bool field from a requests payload
    """
    val = payload.get(key)

    if isinstance(val, bool):
        return val

    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False

    return default


# ------------------ Async bridge helpers ------------------


def do_async_to_sync(async_fn):
    """
    Convert an async function (no args) to sync call.
    Prefers asgiref, which is the most reliable for sync Flask under gunicorn.
    """
    try:
        return async_to_sync(async_fn)
    except Exception as e:
        utils_logger.error(f"Exception path in _async_to_sync, details: {e}")
        # fallback: run in a fresh event loop (ok for dev; not ideal at scale)
        import asyncio

        def _runner():
            """Run ``async_fn`` to completion in a fresh event loop."""
            return asyncio.run(async_fn())

        return _runner
