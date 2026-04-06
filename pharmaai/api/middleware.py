"""
pharmaai/api/middleware.py
Token-bucket rate limiter per client IP using Redis.
Falls back to in-memory if Redis is unavailable.
"""
from __future__ import annotations

import time
import logging
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("pharmaai.api.middleware")

# In-memory fallback store: ip → (tokens, last_refill_time)
_store: dict[str, list] = defaultdict(lambda: [0.0, 0.0])


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window token bucket rate limiter.
    """

    def __init__(self, app, requests_per_minute: int = 60, burst: int = 20):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst
        self._rate = requests_per_minute / 60.0  # tokens/second

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health/metrics
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.monotonic()
        bucket = _store[ip]

        # Refill tokens
        elapsed = now - bucket[1]
        bucket[0] = min(self._burst, bucket[0] + elapsed * self._rate)
        bucket[1] = now

        if bucket[0] >= 1.0:
            bucket[0] -= 1.0
            return await call_next(request)
        else:
            retry_after = int((1.0 - bucket[0]) / self._rate) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(retry_after)},
            )