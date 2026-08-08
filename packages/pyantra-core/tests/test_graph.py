"""Tests for basic linear graph execution (sync and async)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from conftest import State

from pyantra import END, Graph, RunStatus


@dataclass
class Counter:
    value: int


def test_linear_sync() -> None:
    graph = Graph(Counter)

    @graph.node
    def increment(state: Counter) -> Counter:
        state.value += 1
        return state

    @graph.node
    def double(state: Counter) -> Counter:
        state.value *= 2
        return state

    graph.set_entry_point(increment)
    graph.add_edge(increment, double)

    app = graph.compile()
    result = app.run(Counter(value=1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.value == 4


async def test_linear_async() -> None:
    graph = Graph(Counter)

    @graph.node
    def increment(state: Counter) -> Counter:
        state.value += 1
        return state

    @graph.node
    def double(state: Counter) -> Counter:
        state.value *= 2
        return state

    graph.set_entry_point(increment)
    graph.add_edge(increment, double)

    result = await graph.compile().arun(Counter(value=1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.value == 4


async def test_async_node_is_awaited() -> None:
    graph = Graph(Counter)

    @graph.node
    async def increment(state: Counter) -> Counter:
        state.value += 1
        return state

    @graph.node
    async def double(state: Counter) -> Counter:
        state.value *= 2
        return state

    graph.set_entry_point(increment)
    graph.add_edge(increment, double)

    result = await graph.compile().arun(Counter(value=1))

    assert result.state is not None
    assert result.state.value == 4


def test_in_place_mutation_returns_none() -> None:
    graph = Graph(Counter)

    @graph.node
    def increment(state: Counter) -> None:
        state.value += 1

    @graph.node
    def double(state: Counter) -> Counter:
        state.value *= 2
        return state

    graph.set_entry_point(increment)
    graph.add_edge(increment, double)

    result = graph.compile().run(Counter(value=1))

    assert result.state is not None
    assert result.state.value == 4


def test_explicit_end_sentinel() -> None:
    graph = Graph(Counter)

    @graph.node
    def increment(state: Counter) -> Counter:
        state.value += 1
        return state

    graph.set_entry_point(increment)
    graph.add_edge(increment, END)

    result = graph.compile().run(Counter(value=1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.value == 2


def test_node_decorator_with_name(graph: Graph[State]) -> None:
    @graph.node(name="custom_name")
    def fn(state: State) -> State:
        state.history.append("ran")
        return state

    assert fn.name == "custom_name"

    graph.set_entry_point("custom_name")
    result = graph.compile().run(State())

    assert result.state is not None
    assert result.state.history == ["ran"]


def test_add_node_with_explicit_name(graph: Graph[State]) -> None:
    node = graph.add_node(lambda state: state, name="lambdanode")
    graph.set_entry_point(node)

    result = graph.compile().run(State())

    assert result.status == RunStatus.COMPLETED


def test_terminal_node_implicit_end() -> None:
    graph = Graph(Counter)

    @graph.node
    def increment(state: Counter) -> Counter:
        state.value += 1
        return state

    graph.set_entry_point(increment)

    result = graph.compile().run(Counter(value=1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.value == 2


@pytest.mark.parametrize("value", [2, 10])
def test_run_ids_are_unique(value: int) -> None:
    graph = Graph(Counter)

    @graph.node
    def increment(state: Counter) -> Counter:
        state.value += 1
        return state

    graph.set_entry_point(increment)
    app = graph.compile()

    first = app.run(Counter(value=value))
    second = app.run(Counter(value=value))

    assert first.run_id != second.run_id
    assert len(first.run_id) == 32
