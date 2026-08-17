"""In-memory cache backend."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pyantra_memory.cache.base import CacheBackend, CacheRegistry


@dataclass
class _Entry:
    """A cached value with an optional expiry timestamp."""

    value: bytes
    expires_at: float | None = None


class InMemoryCache(CacheBackend):
    """A simple in-process cache backed by a dictionary.

    Not suitable for distributed systems or processes that need to share
    state. Ideal for testing and single-process workflows.
    """

    def __init__(self) -> None:
        self._store: dict[str, _Entry] = {}

    def get(self, key: str) -> bytes | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        expires_at = (time.monotonic() + ttl) if ttl is not None else None
        self._store[key] = _Entry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


CacheRegistry.register("memory", InMemoryCache)

__all__ = ["InMemoryCache"]
