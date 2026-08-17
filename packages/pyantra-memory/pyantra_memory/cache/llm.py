"""LLM caching wrapper."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pyantra.llm.types import LLMResponse, Message, Usage
from pyantra_memory.cache.base import CacheBackend


def _cache_key(messages: Sequence[Message], kwargs: dict[str, Any]) -> str:
    """Derive a deterministic cache key from messages and kwargs."""
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "kwargs": kwargs,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _serialize_response(resp: LLMResponse) -> bytes:
    """Serialize an LLMResponse to bytes for caching."""
    data = {
        "content": resp.content,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_tokens": resp.usage.cache_tokens,
            "cost": resp.usage.cost,
            "model": resp.usage.model,
        },
    }
    return json.dumps(data).encode()


def _deserialize_response(data: bytes) -> LLMResponse:
    """Deserialize bytes back into an LLMResponse."""
    obj = json.loads(data)
    usage = Usage(
        input_tokens=obj["usage"]["input_tokens"],
        output_tokens=obj["usage"]["output_tokens"],
        cache_tokens=obj["usage"]["cache_tokens"],
        cost=obj["usage"]["cost"],
        model=obj["usage"]["model"],
    )
    return LLMResponse(content=obj["content"], usage=usage)


class CachedLLM:
    """A wrapper around any LLM that caches responses.

    On a cache hit, returns the cached response with zero-cost ``Usage``
    so the run's cost tracking stays accurate (cached calls are free).

    Integrates with :class:`~pyantra.runtime.context.RunContext` when
    available, emitting ``cache.hit`` and ``cache.miss`` events.

    Example::

        real_llm = GroqLLM(model="llama-3")
        cached = CachedLLM(llm=real_llm, backend=InMemoryCache(), ttl=3600)
        response = cached.generate(messages)  # checks cache first
    """

    def __init__(
        self,
        llm: Any,
        backend: CacheBackend,
        ttl: float | None = None,
    ) -> None:
        self._llm = llm
        self._backend = backend
        self._ttl = ttl

    def _emit(self, event: str) -> None:
        """Emit a cache event to the active RunContext if available."""
        try:
            from pyantra.runtime.context import run_context

            ctx = run_context.get()
            if ctx is not None:
                ctx.emit(event, message="cached_llm")
        except Exception:
            pass

    def generate(self, messages: Sequence[Message], **kwargs: object) -> LLMResponse:
        key = _cache_key(messages, dict(kwargs))
        cached = self._backend.get(key)
        if cached is not None:
            self._emit("cache.hit")
            resp = _deserialize_response(cached)
            return LLMResponse(
                content=resp.content,
                usage=Usage(),  # zero-cost on cache hit
            )

        self._emit("cache.miss")
        resp = self._llm.generate(messages, **kwargs)
        self._backend.set(key, _serialize_response(resp), ttl=self._ttl)
        return resp

    async def agenerate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        key = _cache_key(messages, dict(kwargs))
        cached = self._backend.get(key)
        if cached is not None:
            self._emit("cache.hit")
            resp = _deserialize_response(cached)
            return LLMResponse(
                content=resp.content,
                usage=Usage(),  # zero-cost on cache hit
            )

        self._emit("cache.miss")
        resp = await self._llm.agenerate(messages, **kwargs)
        self._backend.set(key, _serialize_response(resp), ttl=self._ttl)
        return resp


__all__ = ["CachedLLM"]
