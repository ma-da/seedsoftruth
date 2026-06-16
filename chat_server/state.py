"""Shared retrieval-engine state.

Owns the process-wide ``RetrievalState`` produced by
``rag_controller.boot()`` and the lazy initializer used to build it.

Both the request handlers and the background worker read the state
through ``get_retrieval_state()`` rather than importing a module global
directly: a plain ``from state import retrieval_state`` would bind the
name once, at import time, and never observe the value that
``init_state()`` later assigns. The accessor resolves the live value on
every call, so all modules see the same, current state.
"""

import threading
import time

import config
import logging_config
import rag_controller

state_logger = logging_config.get_logger("state")

_retrieval_state = None
_state_lock = threading.Lock()
_last_init_attempt_ts = 0.0
_last_init_error = None


def get_retrieval_state():
    """Return the current RetrievalState, or None if not yet initialized."""
    return _retrieval_state


def get_last_init_error():
    """Return the error message from the most recent failed init, or None."""
    return _last_init_error


def init_state(force: bool = False) -> bool:
    """Initialize the retrieval state by calling ``rag_controller.boot()``.

    Uses a lock to avoid concurrent boot calls and a cooldown
    (``config.INIT_RETRY_COOLDOWN_S``) to avoid spamming ``boot()`` after
    a failure. Idempotent: returns immediately if state is already built.

    Args:
        force: When True, bypass the retry cooldown.

    Returns:
        True if the retrieval state is ready, False otherwise.
    """
    global _retrieval_state, _last_init_attempt_ts, _last_init_error

    if _retrieval_state is not None:
        return True

    now = time.time()
    if (not force) and (
        now - _last_init_attempt_ts
    ) < config.INIT_RETRY_COOLDOWN_S:
        return False

    with _state_lock:
        if _retrieval_state is not None:
            return True

        now = time.time()
        if (not force) and (
            now - _last_init_attempt_ts
        ) < config.INIT_RETRY_COOLDOWN_S:
            return False

        _last_init_attempt_ts = now
        _last_init_error = None

        state_logger.info("init_state: starting rag_controller.boot()...")

        try:
            _retrieval_state = rag_controller.boot()
            if _retrieval_state is None:
                _last_init_error = "boot() returned None"
                state_logger.error(_last_init_error)
                return False

            state_logger.info("init_state: ✅ retrieval_state initialized")
            return True

        except Exception as e:
            _last_init_error = f"boot() failed: {e}"
            state_logger.exception(_last_init_error)
            _retrieval_state = None
            return False


def ensure_state() -> bool:
    """Lazily (re)initialize retrieval state for a request. True if ready."""
    return init_state(force=False)
