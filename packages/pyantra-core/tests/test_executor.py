"""Tests for the Run object and executor behavior."""

from __future__ import annotations

import pytest
from conftest import State

from pyantra import Graph, NodeExecutionError, RunStatus


def test_run_status_is_pending_then_completed() -> None:
    graph = Graph(State)

    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point(a)

    result = graph.compile().run(State())

    assert result.status == RunStatus.COMPLETED


def test_node_failure_marks_run_failed(graph: Graph[State]) -> None:
    @graph.node
    def boom(state: State) -> State:
        raise ValueError("kaboom")

    graph.set_entry_point(boom)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "boom" in result.error
    assert "kaboom" in result.error
    assert isinstance(result.exception, NodeExecutionError)
    assert isinstance(result.exception.__cause__, ValueError)
    assert result.exception.node == "boom"


def test_state_validation_at_run(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point(a)

    result = graph.compile().run({"not": "state"})  # type: ignore[arg-type]

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "State" in result.error


def test_run_events_trace_execution(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    @graph.node
    def b(state: State) -> State:
        return state

    graph.set_entry_point(a)
    graph.add_edge(a, b)

    result = graph.compile().run(State())
    names = [event.event for event in result.events]

    assert names[0] == "run.started"
    assert names[-1] == "run.completed"
    assert "node.started" in names
    assert "node.completed" in names
    assert "edge.selected" in names
    assert result.events[1].node == "a"


def test_node_failed_event_present(graph: Graph[State]) -> None:
    @graph.node
    def boom(state: State) -> State:
        raise RuntimeError("nope")

    graph.set_entry_point(boom)

    result = graph.compile().run(State())
    events = {event.event for event in result.events}

    assert "node.failed" in events
    assert "run.failed" in events


def test_max_iterations_guard() -> None:
    graph = Graph(State)

    @graph.node
    def start(state: State) -> State:
        return state

    @graph.node
    def loop(state: State) -> State:
        state.history.append("x")
        return state

    def route(state: State) -> str:
        return "loop"

    graph.set_entry_point(start)
    graph.add_edge(start, loop)
    graph.add_conditional_edges(loop, route, {"loop": loop})

    result = graph.compile().run(State(), max_iterations=5)

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "max_iterations" in result.error


async def test_max_iterations_guard_async() -> None:
    graph = Graph(State)

    @graph.node
    def loop(state: State) -> State:
        return state

    def route(state: State) -> str:
        return "loop"

    graph.set_entry_point(loop)
    graph.add_conditional_edges(loop, route, {"loop": loop})

    result = await graph.compile().arun(State(), max_iterations=3)

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "max_iterations" in result.error


def test_state_reflects_latest_completed_node(graph: Graph[State]) -> None:
    @graph.node
    def mutate(state: State) -> State:
        state.history.append("mutate")
        return state

    @graph.node
    def boom(state: State) -> State:
        raise RuntimeError("fail")

    graph.set_entry_point(mutate)
    graph.add_edge(mutate, boom)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert result.state is not None
    assert result.state.history == ["mutate"]


def test_wrong_return_type_fails_run(graph: Graph[State]) -> None:
    @graph.node
    def bad(state: State) -> State:
        return "not state"  # type: ignore[return-value]

    graph.set_entry_point(bad)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "State" in result.error


def test_run_sync_without_running_loop(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        state.value += 1
        return state

    graph.set_entry_point(a)

    result = graph.compile().run_sync(State(value=1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.value == 2


async def test_run_inside_running_loop_raises(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point(a)

    with pytest.raises(RuntimeError, match="running event loop"):
        graph.compile().run(State())


async def test_run_sync_inside_running_loop(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        state.value += 1
        return state

    graph.set_entry_point(a)

    result = graph.compile().run_sync(State(value=1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.value == 2
