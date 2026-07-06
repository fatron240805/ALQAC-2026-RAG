"""Enforce the competition's '1 API request / 5 seconds' constraint."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Blocking rate limiter: wait() sleeps just enough to respect min_interval."""

    def __init__(self, min_interval_seconds: float = 5.0):
        self.min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._last_call_at: float | None = None
        self.call_count = 0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call_at is not None:
                elapsed = now - self._last_call_at
                remaining = self.min_interval_seconds - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            self._last_call_at = time.monotonic()
            self.call_count += 1
