"""
Contains helpful utilities for use by other files
"""
import time
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


class SimpleUserRateLimiter:
    """
    A class that tracks rate limiting per user. Only allows one request per X seconds.
    """
    interval_secs = 0
    user_data = {}

    def __init__(self, interval_secs_param):
        if not isinstance(interval_secs_param, int):
            raise RuntimeError("interval_secs_param must be int")
        if interval_secs_param <= 0:
            raise RuntimeError("interval_secs_param must be greater than zero")
        self.interval_secs = interval_secs_param

    def check(self, user_id) -> True:
        """
        Check if current user_id operation is allowed. Tracks current time versus last time.
        Returns true if time difference greater than interval, false if less
        """
        if not isinstance(user_id, str):
            raise RuntimeError("user_id must be str")
        if len(user_id) == 0:
            raise RuntimeError("user_id must not be empty")

        curr_time = int(time.time())
        last_time = self.user_data.get(user_id)
        self.user_data[user_id] = curr_time

        if last_time is None:
            return True
        else:
            assert curr_time >= last_time  # if not true, then its nonsensical
            diff = curr_time - last_time
            return diff >= self.interval_secs
