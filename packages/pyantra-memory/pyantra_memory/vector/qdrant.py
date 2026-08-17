"""Qdrant vector store backend."""

from __future__ import annotations

import uuid
from typing import Any

from pyantra_memory.vector.base import ScoredResult, VectorRegistry, VectorStore

_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class QdrantVectorStore(VectorStore):
    """Vector store backed by a Qdrant server or in-memory instance.

    Requires the ``qdrant-client`` package::

        pip install pyantra-memory[qdrant]

    Example::

        store = QdrantVectorStore(collection="my-docs", url="http://localhost:6333")
        store.add(["doc1"], [[0.1, 0.2, 0.3]], [{"source": "web"}])
        results = store.query([0.1, 0.2, 0.3], k=5)
    """

    def __init__(
        self,
        collection: str = "pyantra",
        vector_size: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install pyantra-memory[qdrant]"
            ) from exc

        self._collection = collection
        self._vector_size = vector_size
        self._id_map: dict[uuid.UUID, str] = {}

        if url is not None:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(location=":memory:")

        if vector_size is not None and not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    @staticmethod
    def _to_uuid(id_: str) -> uuid.UUID:
        """Deterministically convert a string ID to a UUID."""
        return uuid.uuid5(_NAMESPACE, id_)

    def _ensure_collection(self, vector_size: int) -> None:
        """Create collection on first add if vector_size was not provided at init."""
        if self._vector_size is None:
            from qdrant_client.models import Distance, VectorParams

            self._vector_size = vector_size
            if not self._client.collection_exists(self._collection):
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        from qdrant_client.models import PointStruct

        if vectors:
            self._ensure_collection(len(vectors[0]))

        points = []
        for id_, vec, meta in zip(ids, vectors, metadatas, strict=True):
            uuid_id = self._to_uuid(id_)
            self._id_map[uuid_id] = id_
            points.append(
                PointStruct(id=uuid_id, vector=vec, payload=meta)
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def query(
        self,
        vector: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[ScoredResult]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter: Filter | None = None
        if filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=val))
                    for key, val in filter.items()
                ]
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=k,
        )

        return [
            ScoredResult(
                id=self._id_map.get(
                    point.id if isinstance(point.id, uuid.UUID)
                    else uuid.UUID(str(point.id)),
                    str(point.id),
                ),
                score=point.score,
                metadata=point.payload or {},
            )
            for point in response.points
        ]

    def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        uuid_ids = []
        for id_ in ids:
            uuid_id = self._to_uuid(id_)
            self._id_map.pop(uuid_id, None)
            uuid_ids.append(uuid_id)

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=uuid_ids),
        )

    def count(self) -> int:
        result = self._client.count(collection_name=self._collection)
        return result.count


VectorRegistry.register("qdrant", QdrantVectorStore)

__all__ = ["QdrantVectorStore"]
