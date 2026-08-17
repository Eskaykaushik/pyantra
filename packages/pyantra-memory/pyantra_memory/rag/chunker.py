"""Text chunking strategies for RAG pipelines."""

from __future__ import annotations

import re
from typing import Protocol


class TextChunker(Protocol):
    """Protocol for text chunking strategies.

    Any object that implements ``chunk(text) -> list[str]`` can be used
    as a chunker in the RAG pipeline.
    """

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks."""
        ...


class FixedSizeChunker:
    """Split text into fixed-size chunks with optional overlap.

    Example::

        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk("long document text...")
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        step = self._chunk_size - self._overlap
        chunks: list[str] = []
        for i in range(0, len(text), step):
            chunk = text[i : i + self._chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks


class SentenceChunker:
    """Split text into chunks by sentence boundaries.

    Attempts to keep chunks under ``max_chunk_size`` by grouping
    consecutive sentences. Falls back to splitting individual
    long sentences by words.

    Example::

        chunker = SentenceChunker(max_chunk_size=1000)
        chunks = chunker.chunk("First sentence. Second sentence.")
    """

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chunk_size: int = 1000) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self._max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = self._SENTENCE_RE.split(text.strip())
        sentences = [s for s in sentences if s]
        if not sentences:
            return [text]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current and current_len + 1 + sentence_len > self._max_chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            if sentence_len > self._max_chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                word_chunks = self._split_long_sentence(sentence)
                chunks.extend(word_chunks)
            else:
                current.append(sentence)
                current_len += sentence_len + (1 if current_len > 0 else 0)

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _split_long_sentence(self, sentence: str) -> list[str]:
        words = sentence.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            word_len = len(word)
            if current and current_len + 1 + word_len > self._max_chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += word_len + (1 if current_len > 0 else 0)

        if current:
            chunks.append(" ".join(current))
        return chunks


class RecursiveChunker:
    """Recursively split text by separators, falling back to smaller delimiters.

    Tries splitting by paragraph (``\\n\\n``), then by line (``\\n``),
    then by sentence, then by word. Each resulting chunk is at most
    ``max_chunk_size`` characters.

    Example::

        chunker = RecursiveChunker(max_chunk_size=500)
        chunks = chunker.chunk("paragraph one\\n\\nparagraph two")
    """

    _SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, max_chunk_size: int = 500) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self._max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split_recursive(text, self._SEPARATORS)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self._max_chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        rest = separators[1:]
        parts = text.split(sep)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self._max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) <= self._max_chunk_size:
                    current = part
                elif rest:
                    chunks.extend(self._split_recursive(part, rest))
                    current = ""
                else:
                    chunks.extend(self._hard_split(part))
                    current = ""

        if current and current.strip():
            chunks.append(current)
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        return [
            text[i : i + self._max_chunk_size]
            for i in range(0, len(text), self._max_chunk_size)
            if text[i : i + self._max_chunk_size]
        ]


__all__ = [
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "TextChunker",
]
