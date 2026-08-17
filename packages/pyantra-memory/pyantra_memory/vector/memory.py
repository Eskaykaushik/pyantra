"""In-memory vector store backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pyantra_memory.vector.base import ScoredResult, VectorRegistry, VectorStore


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: list[float]) -> float:
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in a))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


@dataclass
class _Entry:
    id: str
    vector: list[float]
    metadata: dict[str, Any]


class InMemoryVectorStore(VectorStore):
    """A simple in-process vector store using brute-force cosine search.

    Not suitable for large-scale production use. Ideal for testing,
    prototyping, and small datasets where an external service is
    overkill.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for id_, vec, meta in zip(ids, vectors, metadatas, strict=True):
            self._entries[id_] = _Entry(id=id_, vector=vec, metadata=meta)

    def query(
        self,
        vector: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredResult]:
        candidates = list(self._entries.values())
        if filter:
            candidates = [
                e
                for e in candidates
                if all(e.metadata.get(k_) == v for k_, v in filter.items())
            ]

        scored = [
            (e.id, _cosine_similarity(vector, e.vector), e.metadata)
            for e in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            ScoredResult(id=id_, score=score, metadata=meta)
            for id_, score, meta in scored[:k]
        ]

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            self._entries.pop(id_, None)

    def count(self) -> int:
        return len(self._entries)


VectorRegistry.register("memory", InMemoryVectorStore)

__all__ = ["InMemoryVectorStore"]
