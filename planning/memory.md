# pyantra-memory Implementation Plan

> **Important: Execute one file per step. Verify after each step before moving to the next.**

## Overview

Implement `pyantra-memory` with 6 features: LLM caching, vector stores, RAG, token budgets, deterministic replay, model tiering.

**Approach:** Incremental — one file at a time, verify after each step.

## Package Structure

```
packages/pyantra-memory/
├── pyproject.toml
├── README.md
├── py.typed
├── pyantra_memory/
│   ├── __init__.py
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── base.py          # CacheBackend ABC + CacheRegistry
│   │   ├── memory.py        # InMemoryCache
│   │   ├── sqlite.py        # SQLiteCache
│   │   └── llm.py           # CachedLLM decorator
│   ├── vector/
│   │   ├── __init__.py
│   │   ├── base.py          # VectorStore ABC + Embedder Protocol + VectorRegistry
│   │   ├── memory.py        # InMemoryVectorStore
│   │   └── qdrant.py        # QdrantVectorStore
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py       # TextChunker strategy protocol + implementations
│   │   └── pipeline.py      # RAGPipeline
│   ├── budget/
│   │   ├── __init__.py
│   │   └── tracker.py       # TokenBudget + BudgetedLLM
│   ├── replay/
│   │   ├── __init__.py
│   │   ├── recorder.py      # TraceRecorder
│   │   └── mock.py          # ReplayLLM
│   └── tier/
│       ├── __init__.py
│       ├── base.py          # ModelRouter ABC + RouterRegistry
│       └── cost.py          # CostBasedRouter
└── tests/
    ├── __init__.py
    ├── test_cache.py
    ├── test_vector.py
    ├── test_rag.py
    ├── test_budget.py
    ├── test_replay.py
    └── test_tier.py
```

## Design Principles

- **Plugin registry** per ABC — self-registering backends
- **Lazy optional imports** — zero core deps, backends are optional
- **Strategy pattern** — chunking, routing, caching all pluggable
- **Composable** — RAGPipeline takes VectorStore + Embedder + LLM as args
- **Follows core patterns** — CheckpointStore ABC, Serializer, RunContext conventions

## Execution Steps

| # | Step | Files | Verify |
|---|------|-------|--------|
| 1 | Update pyproject.toml files | root `pyproject.toml`, memory `pyproject.toml` | `uv sync` |
| 2 | Cache ABC + Registry | `cache/base.py` | `mypy` pass |
| 3 | InMemoryCache | `cache/memory.py`, `tests/test_cache.py` (part 1) | pytest |
| 4 | SQLiteCache | `cache/sqlite.py`, `tests/test_cache.py` (part 2) | pytest |
| 5 | CachedLLM wrapper | `cache/llm.py`, `tests/test_cache.py` (part 3) | pytest |
| 6 | VectorStore + Embedder ABCs + Registry | `vector/base.py` | `mypy` pass |
| 7 | InMemoryVectorStore | `vector/memory.py`, `tests/test_vector.py` (part 1) | pytest |
| 8 | QdrantVectorStore | `vector/qdrant.py`, `tests/test_vector.py` (part 2) | pytest |
| 9 | TextChunker | `rag/chunker.py`, `tests/test_rag.py` (part 1) | pytest |
| 10 | RAGPipeline | `rag/pipeline.py`, `tests/test_rag.py` (part 2) | pytest |
| 11 | TokenBudget + BudgetedLLM | `budget/tracker.py`, `tests/test_budget.py` | pytest |
| 12 | TraceRecorder | `replay/recorder.py`, `tests/test_replay.py` (part 1) | pytest |
| 13 | ReplayLLM | `replay/mock.py`, `tests/test_replay.py` (part 2) | pytest |
| 14 | ModelRouter ABC + Registry | `tier/base.py` | `mypy` pass |
| 15 | CostBasedRouter | `tier/cost.py`, `tests/test_tier.py` | pytest |
| 16 | Public API exports | `__init__.py` | `mypy` + full test suite |
| 17 | Final lint + typecheck | — | `ruff check` + `mypy` |

## Interface Definitions

### CacheBackend (cache/base.py)

```python
class CacheBackend(ABC):
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes, ttl: float | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def clear(self) -> None: ...
```

### VectorStore (vector/base.py)

```python
@dataclass(frozen=True)
class ScoredResult:
    id: str
    score: float
    metadata: dict[str, Any]

class VectorStore(ABC):
    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict[str, Any]]) -> None: ...
    def query(self, vector: list[float], k: int = 5, filter: dict[str, Any] | None = None) -> list[ScoredResult]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

### TokenBudget (budget/tracker.py)

```python
@dataclass
class TokenBudget:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost: float | None = None

class BudgetExceeded(Exception): ...

class BudgetedLLM:
    def __init__(self, llm: LLM, budget: TokenBudget) -> None: ...
```

### ModelRouter (tier/base.py)

```python
class ModelRouter(ABC):
    def route(self, messages: Sequence[Message], **kwargs: object) -> str: ...
```

### TraceRecorder / ReplayLLM (replay/)

```python
class TraceRecorder:
    def __init__(self, llm: LLM) -> None: ...
    def get_trace(self) -> list[dict[str, Any]]: ...

class ReplayLLM:
    def __init__(self, trace: list[dict[str, Any]]) -> None: ...
```

### RAGPipeline (rag/pipeline.py)

```python
@dataclass(frozen=True)
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

class TextChunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...

class RAGPipeline:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, llm: LLM, chunker: TextChunker | None = None) -> None: ...
    async def aingest(self, documents: list[Document]) -> int: ...
    async def agenerate(self, query: str, k: int = 5, system_prompt: str | None = None) -> LLMResponse: ...
```

## Dependencies

```toml
[project]
dependencies = ["pyantra>=0.5.1"]

[project.optional-dependencies]
qdrant = ["qdrant-client>=1.9"]
all = ["qdrant-client>=1.9"]
```

## Status

- [ ] Step 1: pyproject.toml updates
- [x] Step 2: cache/base.py
- [x] Step 3: cache/memory.py
- [x] Step 4: cache/sqlite.py
- [x] Step 5: cache/llm.py
- [x] Step 6: vector/base.py
- [x] Step 7: vector/memory.py
- [ ] Step 8: vector/qdrant.py
- [ ] Step 9: rag/chunker.py
- [ ] Step 10: rag/pipeline.py
- [ ] Step 11: budget/tracker.py
- [ ] Step 12: replay/recorder.py
- [ ] Step 13: replay/mock.py
- [ ] Step 14: tier/base.py
- [ ] Step 15: tier/cost.py
- [ ] Step 16: __init__.py
- [ ] Step 17: Final lint + typecheck
