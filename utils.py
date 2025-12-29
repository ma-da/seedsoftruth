"""
Contains helpful utilities for use by other files
"""
import threading


def str_to_bool(s: str, strict = True) -> bool:
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


class SafeInt:
    """
    A thread-safe int wrapper.
    Note the atomic.INT in the atomics package doesn't work for 3.13+ currently. This is a replacement.
    """
    val = 0
    lock = threading.Lock()

    def __init__(self, val):
        self.val = val

    def get(self):
        with self.lock:
            return self.val

    def set(self, val):
        with self.lock:
            self.val = val

    def inc(self):
        with self.lock:
            self.val = self.val + 1

    def dec(self):
        with self.lock:
            self.val = self.val - 1
