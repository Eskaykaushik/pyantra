"""Vector store interface, embedder protocol, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ScoredResult:
    """A vector search result with its similarity score."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Embedder(Protocol):
    """Protocol for text embedding providers.

    Any object that implements ``embed(texts) -> list[list[float]]`` can be
    used as an embedder in the RAG pipeline.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors."""
        ...


class VectorStore(ABC):
    """Abstract storage for vector embeddings.

    Implementations handle the specifics of similarity search, indexing,
    and metadata storage. Backends can be in-memory, database-backed
    (Qdrant, Pinecone, ChromaDB), or distributed.
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors with associated metadata."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredResult]:
        """Find the *k* closest vectors to *vector*.

        ``filter`` is an optional metadata filter. The semantics of
        filtering are backend-specific but should support at least
        equality checks on top-level keys.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove vectors by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""


class VectorRegistry:
    """Registry for vector store implementations.

    Example::

        VectorRegistry.register("memory", InMemoryVectorStore)
        store = VectorRegistry.create("memory")
    """

    _stores: dict[str, type[VectorStore]] = {}

    @classmethod
    def register(cls, name: str, store_cls: type[VectorStore]) -> None:
        cls._stores[name] = store_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> VectorStore:
        if name not in cls._stores:
            available = ", ".join(sorted(cls._stores)) or "(none)"
            raise KeyError(
                f"Unknown vector store {name!r}. Available: {available}"
            )
        return cls._stores[name](**kwargs)

    @classmethod
    def list_stores(cls) -> list[str]:
        return sorted(cls._stores)


__all__ = ["Embedder", "ScoredResult", "VectorRegistry", "VectorStore"]
