"""Cache storage interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    """Abstract storage for key-value cache entries.

    Implementations handle serialization internally — callers store and
    retrieve raw ``bytes``. Backends may be in-memory, on-disk, or
    distributed (Redis, Memcached, etc.).
    """

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Retrieve a cached value by key, or None if missing/expired."""

    @abstractmethod
    def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        """Store a value with an optional TTL in seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single entry."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""


class CacheRegistry:
    """Registry for cache backend implementations.

    Backends self-register via ``CacheRegistry.register(name, cls)`` so
    callers can instantiate them by name without importing the module
    directly.

    Example::

        CacheRegistry.register("memory", InMemoryCache)
        backend = CacheRegistry.create("memory")
    """

    _backends: dict[str, type[CacheBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: type[CacheBackend]) -> None:
        """Register a backend class under *name*."""
        cls._backends[name] = backend_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> CacheBackend:
        """Instantiate a registered backend by *name*."""
        if name not in cls._backends:
            available = ", ".join(sorted(cls._backends)) or "(none)"
            raise KeyError(
                f"Unknown cache backend {name!r}. Available: {available}"
            )
        return cls._backends[name](**kwargs)

    @classmethod
    def list_backends(cls) -> list[str]:
        """Return sorted names of all registered backends."""
        return sorted(cls._backends)


__all__ = ["CacheBackend", "CacheRegistry"]
