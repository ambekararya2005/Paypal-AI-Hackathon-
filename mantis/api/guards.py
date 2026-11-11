"""Abuse guards for the public deployment: per-IP rate limits and request size.

The daily caps and the kill switch live in :mod:`mantis.core.quota`, because the
LLM and PayPal clients must consult them before any live call. This module guards
the HTTP edge:

* **60 writes per IP per minute** (POST and friends: anything that can spend), and
  **300 reads per IP per minute** (GETs: cheap, in-memory, and what the dashboard and
  the playground poll — a 60/min ceiling would have cut off a single open tab);
* **request bodies over 32 KB are refused** (``413``) before they are parsed;
* :class:`SlidingWindow` is reused by the playground for its own **5 runs per IP
  per hour**.

The client IP is the first ``X-Forwarded-For`` hop when present — Render's proxy
sets it — and the socket peer otherwise. In-memory and per process: Render runs
one instance of this service, and a restart forgetting a minute of request
history is harmless where a forgotten *day* of spend would not be (that is why
the caps are on disk and these are not).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Final

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

__all__ = ["MAX_BODY_BYTES", "GuardMiddleware", "SlidingWindow", "client_ip"]

MAX_BODY_BYTES: Final[int] = 32 * 1024

#: Requests that do not count against the limit: the health probe Render polls.
_EXEMPT: Final[tuple[str, ...]] = ("/health",)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SlidingWindow:
    """At most ``limit`` events per ``window_s`` seconds per key. Thread-safe."""

    def __init__(self, limit: int, window_s: float, *, max_keys: int = 50_000) -> None:
        self.limit = limit
        self.window_s = window_s
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """``(allowed, retry_after_seconds)``. Records the hit only when allowed."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, self.window_s - (now - hits[0])
            hits.append(now)
            if len(self._hits) > self.max_keys:  # bound memory under a key flood
                for stale in [k for k, v in self._hits.items() if not v][: self.max_keys // 10]:
                    self._hits.pop(stale, None)
            return True, 0.0

    def remaining(self, key: str, *, now: float | None = None) -> int:
        """How many more hits ``key`` may make right now. Read-only: records nothing.

        It neither appends a hit nor creates a deque for an unseen key, so asking
        cannot consume a slot or grow the table.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return self.limit
            live = sum(1 for t in hits if now - t < self.window_s)
            return max(0, self.limit - live)


def _rate_per_min() -> int:
    raw = os.environ.get("MANTIS_RATE_PER_MIN", "").strip()
    return int(raw) if raw.isdigit() else 60


class GuardMiddleware(BaseHTTPMiddleware):
    """Body-size limit and per-IP rate limit, applied to the whole API."""

    def __init__(self, app: object, *, per_min: int | None = None,
                 reads_per_min: int = 300) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.window = SlidingWindow(per_min or _rate_per_min(), 60.0)
        self.reads = SlidingWindow(reads_per_min, 60.0)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path.endswith(_EXEMPT):
            return await call_next(request)
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": f"request body over {MAX_BODY_BYTES} bytes"},
                                status_code=413)
        window = self.reads if request.method in ("GET", "HEAD") else self.window
        allowed, retry = window.allow(client_ip(request))
        if not allowed:
            kind = "reads" if window is self.reads else "requests"
            return JSONResponse(
                {"detail": f"rate limit: {window.limit} {kind} per minute per IP"},
                status_code=429, headers={"Retry-After": str(max(1, int(retry)))},
            )
        return await call_next(request)
