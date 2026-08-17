"""Tests for vector store backends."""

from __future__ import annotations

from pyantra_memory.vector.base import VectorRegistry
from pyantra_memory.vector.memory import InMemoryVectorStore
from pyantra_memory.vector.qdrant import QdrantVectorStore


class TestInMemoryVectorStore:
    def test_add_and_count(self) -> None:
        store = InMemoryVectorStore()
        store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{"tag": "x"}, {"tag": "y"}])
        assert store.count() == 2

    def test_query_basic(self) -> None:
        store = InMemoryVectorStore()
        store.add(
            ["a", "b", "c"],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [{"t": 1}, {"t": 2}, {"t": 3}],
        )
        results = store.query([1.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].score > results[1].score

    def test_query_with_filter(self) -> None:
        store = InMemoryVectorStore()
        store.add(
            ["a", "b", "c"],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [{"cat": "x"}, {"cat": "y"}, {"cat": "x"}],
        )
        results = store.query([1.0, 0.0], k=10, filter={"cat": "x"})
        assert len(results) == 2
        assert all(r.metadata["cat"] == "x" for r in results)

    def test_delete(self) -> None:
        store = InMemoryVectorStore()
        store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{}, {}])
        store.delete(["a"])
        assert store.count() == 1
        results = store.query([1.0, 0.0], k=5)
        assert len(results) == 1
        assert results[0].id == "b"

    def test_delete_missing(self) -> None:
        store = InMemoryVectorStore()
        store.delete(["nonexistent"])

    def test_overwrite(self) -> None:
        store = InMemoryVectorStore()
        store.add(["a"], [[1.0, 0.0]], [{"v": 1}])
        store.add(["a"], [[0.0, 1.0]], [{"v": 2}])
        assert store.count() == 1
        results = store.query([0.0, 1.0], k=1)
        assert results[0].id == "a"
        assert results[0].metadata["v"] == 2

    def test_empty_query(self) -> None:
        store = InMemoryVectorStore()
        results = store.query([1.0, 0.0], k=5)
        assert results == []

    def test_zero_vector(self) -> None:
        store = InMemoryVectorStore()
        store.add(["a"], [[0.0, 0.0]], [{}])
        results = store.query([1.0, 0.0], k=1)
        assert results[0].score == 0.0


class TestVectorRegistry:
    def test_register_and_create(self) -> None:
        VectorRegistry.register("memory", InMemoryVectorStore)
        store = VectorRegistry.create("memory")
        assert isinstance(store, InMemoryVectorStore)

    def test_list_stores(self) -> None:
        assert "memory" in VectorRegistry.list_stores()

    def test_unknown_store(self) -> None:
        try:
            VectorRegistry.create("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestQdrantVectorStore:
    def test_add_and_count(self) -> None:
        store = QdrantVectorStore(vector_size=2)
        store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{"tag": "x"}, {"tag": "y"}])
        assert store.count() == 2

    def test_query_basic(self) -> None:
        store = QdrantVectorStore(vector_size=2)
        store.add(
            ["a", "b", "c"],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [{"t": 1}, {"t": 2}, {"t": 3}],
        )
        results = store.query([1.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].score > results[1].score

    def test_query_with_filter(self) -> None:
        store = QdrantVectorStore(vector_size=2)
        store.add(
            ["a", "b", "c"],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [{"cat": "x"}, {"cat": "y"}, {"cat": "x"}],
        )
        results = store.query([1.0, 0.0], k=10, filter={"cat": "x"})
        assert len(results) == 2
        assert all(r.metadata["cat"] == "x" for r in results)

    def test_delete(self) -> None:
        store = QdrantVectorStore(vector_size=2)
        store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{}, {}])
        store.delete(["a"])
        assert store.count() == 1
        results = store.query([1.0, 0.0], k=5)
        assert len(results) == 1
        assert results[0].id == "b"

    def test_delete_missing(self) -> None:
        store = QdrantVectorStore(vector_size=2)
        store.delete(["nonexistent"])

    def test_overwrite(self) -> None:
        store = QdrantVectorStore(vector_size=2)
        store.add(["a"], [[1.0, 0.0]], [{"v": 1}])
        store.add(["a"], [[0.0, 1.0]], [{"v": 2}])
        assert store.count() == 1
        results = store.query([0.0, 1.0], k=1)
        assert results[0].id == "a"
        assert results[0].metadata["v"] == 2

    def test_empty_query(self) -> None:
        store = QdrantVectorStore(vector_size=2)
        results = store.query([1.0, 0.0], k=5)
        assert results == []

    def test_auto_create_collection(self) -> None:
        store = QdrantVectorStore()
        store.add(["a"], [[1.0, 0.0, 0.0]], [{}])
        assert store._vector_size == 3

    def test_qdrant_in_registry(self) -> None:
        assert "qdrant" in VectorRegistry.list_stores()
