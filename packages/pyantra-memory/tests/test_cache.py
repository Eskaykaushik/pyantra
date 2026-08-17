"""Tests for cache backends."""

from __future__ import annotations

import time

from pyantra.llm.types import LLMResponse, Message, Usage
from pyantra_memory.cache.memory import InMemoryCache
from pyantra_memory.cache.sqlite import SQLiteCache
from pyantra_memory.cache.base import CacheRegistry
from pyantra_memory.cache.llm import CachedLLM


class TestInMemoryCache:
    def test_get_set(self) -> None:
        cache = InMemoryCache()
        assert cache.get("key") is None
        cache.set("key", b"value")
        assert cache.get("key") == b"value"

    def test_overwrite(self) -> None:
        cache = InMemoryCache()
        cache.set("key", b"old")
        cache.set("key", b"new")
        assert cache.get("key") == b"new"

    def test_delete(self) -> None:
        cache = InMemoryCache()
        cache.set("key", b"value")
        cache.delete("key")
        assert cache.get("key") is None

    def test_delete_missing(self) -> None:
        cache = InMemoryCache()
        cache.delete("nonexistent")

    def test_clear(self) -> None:
        cache = InMemoryCache()
        cache.set("a", b"1")
        cache.set("b", b"2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_ttl_expiry(self) -> None:
        cache = InMemoryCache()
        cache.set("key", b"value", ttl=0.01)
        assert cache.get("key") == b"value"
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_no_ttl(self) -> None:
        cache = InMemoryCache()
        cache.set("key", b"value")
        assert cache.get("key") == b"value"


class TestCacheRegistry:
    def test_register_and_create(self) -> None:
        CacheRegistry.register("memory", InMemoryCache)
        backend = CacheRegistry.create("memory")
        assert isinstance(backend, InMemoryCache)

    def test_list_backends(self) -> None:
        assert "memory" in CacheRegistry.list_backends()

    def test_unknown_backend(self) -> None:
        try:
            CacheRegistry.create("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestSQLiteCache:
    def test_get_set(self, tmp_path: object) -> None:
        cache = SQLiteCache(tmp_path / "test.db")  # type: ignore[arg-type]
        assert cache.get("key") is None
        cache.set("key", b"value")
        assert cache.get("key") == b"value"
        cache.close()

    def test_overwrite(self, tmp_path: object) -> None:
        cache = SQLiteCache(tmp_path / "test.db")  # type: ignore[arg-type]
        cache.set("key", b"old")
        cache.set("key", b"new")
        assert cache.get("key") == b"new"
        cache.close()

    def test_delete(self, tmp_path: object) -> None:
        cache = SQLiteCache(tmp_path / "test.db")  # type: ignore[arg-type]
        cache.set("key", b"value")
        cache.delete("key")
        assert cache.get("key") is None
        cache.close()

    def test_delete_missing(self, tmp_path: object) -> None:
        cache = SQLiteCache(tmp_path / "test.db")  # type: ignore[arg-type]
        cache.delete("nonexistent")
        cache.close()

    def test_clear(self, tmp_path: object) -> None:
        cache = SQLiteCache(tmp_path / "test.db")  # type: ignore[arg-type]
        cache.set("a", b"1")
        cache.set("b", b"2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        cache.close()

    def test_ttl_expiry(self, tmp_path: object) -> None:
        import sqlite3 as _sqlite3

        path = tmp_path / "test.db"  # type: ignore[arg-type]
        cache = SQLiteCache(path)
        cache.set("key", b"value", ttl=0.01)
        row = _sqlite3.connect(str(path)).execute(
            "SELECT expires_at FROM cache WHERE key = ?", ("key",)
        ).fetchone()
        assert row is not None and row[0] is not None
        time.sleep(0.02)
        assert cache.get("key") is None
        cache.close()

    def test_persistence(self, tmp_path: object) -> None:
        path = tmp_path / "test.db"  # type: ignore[arg-type]
        cache1 = SQLiteCache(path)
        cache1.set("key", b"value")
        cache1.close()
        cache2 = SQLiteCache(path)
        assert cache2.get("key") == b"value"
        cache2.close()


class _StubLLM:
    """A stub LLM that records call count."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, messages: Sequence[Message], **kwargs: object) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content="response",
            usage=Usage(input_tokens=10, output_tokens=5, cost=0.01, model="stub"),
        )

    async def agenerate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content="async-response",
            usage=Usage(input_tokens=10, output_tokens=5, cost=0.01, model="stub"),
        )


from collections.abc import Sequence  # noqa: E402


class TestCachedLLM:
    def test_cache_hit(self) -> None:
        llm = _StubLLM()
        cached = CachedLLM(llm=llm, backend=InMemoryCache())
        msgs = [Message(role="user", content="hello")]
        r1 = cached.generate(msgs)
        r2 = cached.generate(msgs)
        assert r1.content == r2.content == "response"
        assert llm.call_count == 1

    def test_cache_miss_different_messages(self) -> None:
        llm = _StubLLM()
        cached = CachedLLM(llm=llm, backend=InMemoryCache())
        r1 = cached.generate([Message(role="user", content="hello")])
        r2 = cached.generate([Message(role="user", content="world")])
        assert r1.content == r2.content == "response"
        assert llm.call_count == 2

    def test_zero_cost_on_hit(self) -> None:
        llm = _StubLLM()
        cached = CachedLLM(llm=llm, backend=InMemoryCache())
        msgs = [Message(role="user", content="hello")]
        cached.generate(msgs)
        cached.generate(msgs)
        resp = cached.generate(msgs)
        assert resp.usage.cost == 0.0
        assert resp.usage.input_tokens == 0

    def test_ttl_propagation(self) -> None:
        llm = _StubLLM()
        cached = CachedLLM(llm=llm, backend=InMemoryCache(), ttl=0.01)
        msgs = [Message(role="user", content="hello")]
        cached.generate(msgs)
        assert llm.call_count == 1
        time.sleep(0.02)
        cached.generate(msgs)
        assert llm.call_count == 2
