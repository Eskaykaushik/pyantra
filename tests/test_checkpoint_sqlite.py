"""Tests for the SQLite-backed checkpoint store."""

from __future__ import annotations

from conftest import State

from pyantra import Graph, RunStatus, SQLiteCheckpointStore


def test_sqlite_resume_after_failure(tmp_path) -> None:
    store: SQLiteCheckpointStore[State] = SQLiteCheckpointStore(
        str(tmp_path / "checkpoints.db")
    )
    calls = {"b": 0}
    graph = Graph(State)

    @graph.node
    def a(state: State) -> State:
        state.history.append("a")
        return state

    @graph.node
    def b(state: State) -> State:
        calls["b"] += 1
        if calls["b"] == 1:
            raise RuntimeError("first failure")
        state.history.append("b")
        return state

    graph.set_entry_point(a)
    graph.add_edge(a, b)

    app = graph.compile()
    run_id = "sqlite-run"

    first = app.run(State(), checkpointer=store, run_id=run_id)

    assert first.status == RunStatus.FAILED
    assert first.state is not None
    assert first.state.history == ["a"]

    second = app.run(State(), checkpointer=store, run_id=run_id)

    assert second.status == RunStatus.COMPLETED
    assert second.state is not None
    assert second.state.history == ["a", "b"]
    store.close()


def test_sqlite_store_survives_new_instance(tmp_path) -> None:
    db = str(tmp_path / "persist.db")
    calls = {"b": 0}
    graph = Graph(State)

    @graph.node
    def a(state: State) -> State:
        state.history.append("a")
        return state

    @graph.node
    def b(state: State) -> State:
        calls["b"] += 1
        if calls["b"] == 1:
            raise RuntimeError("first failure")
        state.history.append("b")
        return state

    graph.set_entry_point(a)
    graph.add_edge(a, b)

    app = graph.compile()
    run_id = "persist-run"

    first_store: SQLiteCheckpointStore[State] = SQLiteCheckpointStore(db)
    first = app.run(State(), checkpointer=first_store, run_id=run_id)
    first_store.close()

    assert first.status == RunStatus.FAILED

    second_store: SQLiteCheckpointStore[State] = SQLiteCheckpointStore(db)
    second = app.run(State(), checkpointer=second_store, run_id=run_id)
    second_store.close()

    assert second.status == RunStatus.COMPLETED
    assert second.state is not None
    assert second.state.history == ["a", "b"]


def test_sqlite_delete_removes_checkpoint(tmp_path) -> None:
    store: SQLiteCheckpointStore[State] = SQLiteCheckpointStore(
        str(tmp_path / "delete.db")
    )
    graph = Graph(State)

    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point(a)

    graph.compile().run(State(), checkpointer=store, run_id="del-run")

    assert store.load("del-run") is None
    store.close()


def test_sqlite_load_missing_returns_none(tmp_path) -> None:
    store: SQLiteCheckpointStore[State] = SQLiteCheckpointStore(
        str(tmp_path / "missing.db")
    )
    assert store.load("nope") is None
    store.close()
