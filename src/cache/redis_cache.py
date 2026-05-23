"""Redis-backed prediction cache with graceful fallback when Redis is unavailable."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def _cache_key(features: dict[str, Any], model: str) -> str:
    payload = json.dumps({"f": features, "m": model}, sort_keys=True)
    return "mlapi:pred:" + hashlib.sha256(payload.encode()).hexdigest()


class RedisCache:
    """Thin wrapper around Redis that never raises — if Redis is down, all ops are no-ops."""

    def __init__(self, url: str, ttl: int) -> None:
        import redis

        self._client = redis.from_url(url, decode_responses=True, socket_connect_timeout=1)
        self._ttl = ttl
        self._available = self._ping()

    def _ping(self) -> bool:
        try:
            self._client.ping()
            log.info("Redis connected")
            return True
        except Exception:
            log.warning("Redis unavailable — caching disabled")
            return False

    def get(self, features: dict[str, Any], model: str) -> dict[str, Any] | None:
        if not self._available:
            return None
        try:
            raw = self._client.get(_cache_key(features, model))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, features: dict[str, Any], model: str, value: dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            self._client.setex(_cache_key(features, model), self._ttl, json.dumps(value))
        except Exception:
            pass
