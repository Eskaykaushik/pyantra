"""Tests for checkpointing and run resume."""

from __future__ import annotations

from conftest import State

from pyantra import Graph, MemoryCheckpointStore, RunStatus


def test_checkpoint_allows_resume_after_failure() -> None:
    store: MemoryCheckpointStore[State] = MemoryCheckpointStore()
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
    run_id = "run-1"

    first = app.run(State(), checkpointer=store, run_id=run_id)

    assert first.status == RunStatus.FAILED
    assert first.state is not None
    assert first.state.history == ["a"]
    checkpoint = store.load(run_id)
    assert checkpoint is not None
    assert checkpoint.resume_at == "b"
    assert checkpoint.state.history == ["a"]

    second = app.run(State(), checkpointer=store, run_id=run_id)

    assert second.status == RunStatus.COMPLETED
    assert second.run_id == run_id
    assert second.state is not None
    assert second.state.history == ["a", "b"]


def test_checkpoint_deleted_after_successful_run() -> None:
    store: MemoryCheckpointStore[State] = MemoryCheckpointStore()
    graph = Graph(State)

    @graph.node
    def a(state: State) -> State:
        state.history.append("a")
        return state

    graph.set_entry_point(a)

    result = graph.compile().run(State(), checkpointer=store, run_id="run-2")

    assert result.status == RunStatus.COMPLETED
    assert store.load("run-2") is None


def test_different_run_id_starts_fresh() -> None:
    store: MemoryCheckpointStore[State] = MemoryCheckpointStore()
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

    app.run(State(), checkpointer=store, run_id="first")
    fresh = app.run(State(), checkpointer=store, run_id="second")

    assert fresh.status == RunStatus.COMPLETED
    assert fresh.state is not None
    assert fresh.state.history == ["a", "b"]


def test_checkpoint_events_are_continued_on_resume() -> None:
    store: MemoryCheckpointStore[State] = MemoryCheckpointStore()
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

    app.run(State(), checkpointer=store, run_id="events-run")
    resumed = app.run(State(), checkpointer=store, run_id="events-run")

    node_starts = [
        e for e in resumed.events if e.event == "node.started" and e.node == "a"
    ]
    assert len(node_starts) == 1
    assert "run.resumed" in {e.event for e in resumed.events}
