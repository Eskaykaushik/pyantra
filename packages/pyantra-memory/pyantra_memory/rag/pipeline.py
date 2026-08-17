"""RAG pipeline: ingest documents, retrieve context, generate answers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from pyantra import LLM, LLMResponse, Message
from pyantra_memory.rag.chunker import RecursiveChunker, TextChunker
from pyantra_memory.vector.base import Embedder, VectorStore


@dataclass(frozen=True)
class Document:
    """A document to ingest into the RAG pipeline."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Retrieval-augmented generation pipeline.

    Ingests documents into a vector store, retrieves relevant chunks
    for a query, and generates an answer using an LLM.

    Example::

        pipeline = RAGPipeline(
            vector_store=store,
            embedder=my_embedder,
            llm=my_llm,
        )
        await pipeline.aingest([Document(content="...")])
        response = await pipeline.agenerate("What is X?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        chunker: TextChunker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm
        self._chunker = chunker or RecursiveChunker()

    async def aingest(self, documents: list[Document]) -> int:
        """Ingest documents into the vector store.

        Returns the number of chunks created.
        """
        all_chunks: list[str] = []
        all_metadatas: list[dict[str, Any]] = []

        for doc in documents:
            chunks = self._chunker.chunk(doc.content)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append(doc.metadata)

        if not all_chunks:
            return 0

        vectors = self._embedder.embed(all_chunks)
        ids = [uuid.uuid4().hex for _ in all_chunks]

        self._vector_store.add(
            ids=ids,
            vectors=vectors,
            metadatas=all_metadatas,
        )
        return len(all_chunks)

    async def agenerate(
        self,
        query: str,
        k: int = 5,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate an answer to a query using retrieved context."""
        query_vector = self._embedder.embed([query])[0]
        results = self._vector_store.query(query_vector, k=k)

        context_parts = [r.metadata.get("content", "") for r in results]
        context = "\n\n".join(filter(None, context_parts))

        if not context:
            context = "(No relevant context found)"

        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        user_content = f"Context:\n{context}\n\nQuestion: {query}"
        messages.append(Message(role="user", content=user_content))

        return await self._llm.agenerate(messages)


__all__ = ["Document", "RAGPipeline"]
