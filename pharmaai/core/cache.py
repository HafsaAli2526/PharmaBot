"""
pharmaai/core/cache.py
Async Redis cache with JSON serialisation for API response caching.
Falls back to a simple in-memory dict if Redis is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger("pharmaai.cache")

_memory_cache: dict[str, str] = {}


def _make_key(prefix: str, *parts: Any) -> str:
    raw = ":".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"pharmaai:{prefix}:{digest}"


class Cache:
    def __init__(self):
        self._redis = None
        self._ttl: int = 3600

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            from pharmaai.core.config import get_settings
            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis.url, decode_responses=True)
            self._ttl = settings.redis.ttl_seconds
            await self._redis.ping()
            logger.info("Redis cache connected.")
        except Exception as exc:
            logger.warning("Redis unavailable (%s) – using in-memory cache.", exc)
            self._redis = None
        return self._redis

    async def get(self, key: str) -> Any | None:
        r = await self._get_redis()
        try:
            if r:
                raw = await r.get(key)
            else:
                raw = _memory_cache.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("Cache get failed: %s", exc)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        raw = json.dumps(value, default=str)
        r = await self._get_redis()
        try:
            if r:
                await r.set(key, raw, ex=ttl or self._ttl)
            else:
                _memory_cache[key] = raw
                # Cap in-memory cache size
                if len(_memory_cache) > 1000:
                    oldest = list(_memory_cache.keys())[:100]
                    for k in oldest:
                        _memory_cache.pop(k, None)
        except Exception as exc:
            logger.debug("Cache set failed: %s", exc)

    async def delete(self, key: str) -> None:
        r = await self._get_redis()
        try:
            if r:
                await r.delete(key)
            else:
                _memory_cache.pop(key, None)
        except Exception:
            pass

    def make_ask_key(self, question: str, domain: str | None) -> str:
        return _make_key("ask", question.lower().strip(), domain or "any")

    def make_search_key(self, query: str, domain: str | None, top_k: int) -> str:
        return _make_key("search", query.lower().strip(), domain or "any", top_k)


cache = Cache()