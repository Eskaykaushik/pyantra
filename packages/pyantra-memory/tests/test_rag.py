"""Tests for RAG chunker implementations and pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from pyantra import LLM, LLMResponse, Message, Usage
from pyantra_memory.rag.chunker import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)
from pyantra_memory.rag.pipeline import Document, RAGPipeline
from pyantra_memory.vector.base import ScoredResult, VectorStore


class TestFixedSizeChunker:
    def test_basic(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10)
        chunks = chunker.chunk("hello world foo bar")
        assert chunks == ["hello worl", "d foo bar"]

    def test_exact_fit(self) -> None:
        chunker = FixedSizeChunker(chunk_size=5)
        chunks = chunker.chunk("hello")
        assert chunks == ["hello"]

    def test_overlap(self) -> None:
        chunker = FixedSizeChunker(chunk_size=5, overlap=2)
        chunks = chunker.chunk("abcdefghij")
        assert chunks[0] == "abcde"
        assert chunks[1] == "defgh"

    def test_empty(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10)
        assert chunker.chunk("") == []

    def test_shorter_than_chunk(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100)
        assert chunker.chunk("hi") == ["hi"]

    def test_invalid_chunk_size(self) -> None:
        try:
            FixedSizeChunker(chunk_size=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_overlap(self) -> None:
        try:
            FixedSizeChunker(chunk_size=5, overlap=5)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestSentenceChunker:
    def test_basic(self) -> None:
        chunker = SentenceChunker(max_chunk_size=20)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_groups_sentences(self) -> None:
        chunker = SentenceChunker(max_chunk_size=100)
        text = "Short. Also short. And another."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_empty(self) -> None:
        chunker = SentenceChunker(max_chunk_size=100)
        assert chunker.chunk("") == []

    def test_single_sentence(self) -> None:
        chunker = SentenceChunker(max_chunk_size=100)
        chunks = chunker.chunk("Just one sentence.")
        assert len(chunks) == 1

    def test_invalid_max_size(self) -> None:
        try:
            SentenceChunker(max_chunk_size=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestRecursiveChunker:
    def test_paragraph_split(self) -> None:
        chunker = RecursiveChunker(max_chunk_size=20)
        text = "paragraph one\n\nparagraph two"
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_falls_back_to_line(self) -> None:
        chunker = RecursiveChunker(max_chunk_size=20)
        text = "line one\nline two\nline three"
        chunks = chunker.chunk(text)
        assert all(len(c) <= 20 for c in chunks)

    def test_falls_back_to_word(self) -> None:
        chunker = RecursiveChunker(max_chunk_size=15)
        text = "this is a long sentence that needs splitting"
        chunks = chunker.chunk(text)
        assert all(len(c) <= 15 for c in chunks)
        assert len(chunks) > 1

    def test_empty(self) -> None:
        chunker = RecursiveChunker(max_chunk_size=10)
        assert chunker.chunk("") == []

    def test_short_text(self) -> None:
        chunker = RecursiveChunker(max_chunk_size=100)
        assert chunker.chunk("hello") == ["hello"]

    def test_invalid_max_size(self) -> None:
        try:
            RecursiveChunker(max_chunk_size=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class _MockVectorStore(VectorStore):
    def __init__(self) -> None:
        self._data: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def add(
        self, ids: list[str], vectors: list[list[float]], metadatas: list[dict[str, Any]]
    ) -> None:
        for id_, vec, meta in zip(ids, vectors, metadatas, strict=True):
            self._data[id_] = (vec, meta)

    def query(
        self, vector: list[float], k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[ScoredResult]:
        return [
            ScoredResult(id=id_, score=1.0, metadata=meta)
            for id_, (_, meta) in list(self._data.items())[:k]
        ]

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            self._data.pop(id_, None)

    def count(self) -> int:
        return len(self._data)


class _MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _MockLLM(LLM):
    def __init__(self) -> None:
        self.last_messages: list[Message] = []

    def generate(self, messages: Sequence[Message], **kwargs: object) -> LLMResponse:
        self.last_messages = list(messages)
        return LLMResponse(
            content="mock answer",
            usage=Usage(input_tokens=10, output_tokens=5, model="mock"),
        )

    async def agenerate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        self.last_messages = list(messages)
        return LLMResponse(
            content="mock answer",
            usage=Usage(input_tokens=10, output_tokens=5, model="mock"),
        )


class TestRAGPipeline:
    async def test_ingest(self) -> None:
        store = _MockVectorStore()
        pipeline = RAGPipeline(
            vector_store=store,
            embedder=_MockEmbedder(),
            llm=_MockLLM(),
        )
        count = await pipeline.aingest(
            [Document(content="hello world", metadata={"src": "test"})]
        )
        assert count >= 1
        assert store.count() >= 1

    async def test_ingest_empty(self) -> None:
        store = _MockVectorStore()
        pipeline = RAGPipeline(
            vector_store=store,
            embedder=_MockEmbedder(),
            llm=_MockLLM(),
        )
        count = await pipeline.aingest([])
        assert count == 0

    async def test_generate(self) -> None:
        store = _MockVectorStore()
        llm = _MockLLM()
        pipeline = RAGPipeline(
            vector_store=store,
            embedder=_MockEmbedder(),
            llm=llm,
        )
        await pipeline.aingest(
            [Document(content="some context", metadata={"content": "the answer"})]
        )
        response = await pipeline.agenerate("What is the answer?")
        assert response.content == "mock answer"
        assert len(llm.last_messages) == 1

    async def test_generate_with_system_prompt(self) -> None:
        llm = _MockLLM()
        pipeline = RAGPipeline(
            vector_store=_MockVectorStore(),
            embedder=_MockEmbedder(),
            llm=llm,
        )
        await pipeline.aingest(
            [Document(content="context", metadata={"content": "info"})]
        )
        await pipeline.agenerate("test", system_prompt="Be helpful.")
        assert llm.last_messages[0].role == "system"
        assert llm.last_messages[0].content == "Be helpful."
