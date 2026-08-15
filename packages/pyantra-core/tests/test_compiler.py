"""Tests for graph compilation and validation."""

from __future__ import annotations

import pytest
from conftest import State

from pyantra import END, Graph, GraphCompileError


def test_missing_entry_point(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    graph.add_edge(a, END)

    with pytest.raises(GraphCompileError, match="entry point"):
        graph.compile()


def test_entry_point_not_registered(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point("ghost")

    with pytest.raises(GraphCompileError, match="ghost"):
        graph.compile()


def test_edge_source_not_registered(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point(a)
    graph.add_edge("ghost", a)

    with pytest.raises(GraphCompileError, match="ghost"):
        graph.compile()


def test_edge_to_unknown_node(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point(a)
    graph.add_edge(a, "ghost")

    with pytest.raises(GraphCompileError, match="ghost"):
        graph.compile()


def test_unreachable_node_is_detected(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    @graph.node
    def orphan(state: State) -> State:
        return state

    graph.set_entry_point(a)
    graph.add_edge(a, END)

    with pytest.raises(GraphCompileError, match="orphan"):
        graph.compile()


def test_ambiguous_out_degree(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    @graph.node
    def b(state: State) -> State:
        return state

    @graph.node
    def c(state: State) -> State:
        return state

    graph.set_entry_point(a)
    graph.add_edge(a, b)
    graph.add_edge(a, c)

    with pytest.raises(GraphCompileError, match="multiple unconditional"):
        graph.compile()


def test_mixed_normal_and_conditional(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    @graph.node
    def b(state: State) -> State:
        return state

    def route(state: State) -> str:
        return "b"

    graph.set_entry_point(a)
    graph.add_edge(a, b)
    graph.add_conditional_edges(a, route, {"b": b})

    with pytest.raises(GraphCompileError, match="both normal and conditional"):
        graph.compile()


def test_duplicate_parallel_fanout_is_rejected(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    @graph.node
    def b(state: State) -> State:
        return state

    @graph.node
    def c(state: State) -> State:
        return state

    graph.set_entry_point(a)
    graph.add_parallel_edges(a, b)
    graph.add_parallel_edges(a, c)

    with pytest.raises(GraphCompileError, match="more than one parallel"):
        graph.compile()


def test_duplicate_node_name(graph: Graph[State]) -> None:
    @graph.node(name="same")
    def a(state: State) -> State:
        return state

    with pytest.raises(GraphCompileError, match="Duplicate node name"):

        @graph.node(name="same")
        def b(state: State) -> State:
            return state


def test_unconditional_cycle_is_detected(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    @graph.node
    def b(state: State) -> State:
        return state

    graph.set_entry_point(a)
    graph.add_edge(a, b)
    graph.add_edge(b, a)

    with pytest.raises(GraphCompileError, match="cycle"):
        graph.compile()


def test_conditional_path_to_unknown_node(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    @graph.node
    def b(state: State) -> State:
        return state

    def route(state: State) -> str:
        return "positive"

    graph.set_entry_point(a)
    graph.add_conditional_edges(a, route, {"positive": "ghost", "negative": b})

    with pytest.raises(GraphCompileError, match="ghost"):
        graph.compile()


def test_default_requires_path_map(graph: Graph[State]) -> None:
    @graph.node
    def a(state: State) -> State:
        return state

    graph.set_entry_point(a)

    def route(state: State) -> str:
        return "whatever"

    graph.add_conditional_edges(a, route, default="other")

    with pytest.raises(GraphCompileError, match="path_map"):
        graph.compile()


def test_conditional_cycle_is_allowed(graph: Graph[State]) -> None:
    @graph.node
    def start(state: State) -> State:
        return state

    @graph.node
    def loop(state: State) -> State:
        state.history.append("loop")
        return state

    @graph.node
    def done(state: State) -> State:
        return state

    def route(state: State) -> str:
        return "again" if len(state.history) < 2 else "done"

    graph.set_entry_point(start)
    graph.add_edge(start, loop)
    graph.add_conditional_edges(loop, route, {"again": loop, "done": done})
    graph.add_edge(done, END)

    app = graph.compile()
    result = app.run(State())

    assert result.status.value == "completed"
    assert result.state is not None
    assert result.state.history == ["loop", "loop"]
