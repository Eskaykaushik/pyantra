"""Tests for parallel fan-out and join execution."""

from __future__ import annotations

import asyncio
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated

import pytest

from pyantra import END, Graph, GraphCompileError, RunStatus


@dataclass
class PState:
    value: int = 0
    results: Annotated[list[str], operator.add] = field(default_factory=list)
    order: list[str] = field(default_factory=list)


def test_parallel_branches_merge_with_reducers() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def branch_a(state: PState) -> dict[str, list[str]]:
        return {"results": ["a"]}

    @graph.node
    def branch_b(state: PState) -> dict[str, list[str]]:
        return {"results": ["b"]}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a, branch_b)

    result = graph.compile().run(PState())

    assert result.status == RunStatus.COMPLETED
    assert sorted(result.state.results) == ["a", "b"]


def test_parallel_join_continues_execution() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def branch_a(state: PState) -> dict[str, list[str]]:
        return {"results": ["a"]}

    @graph.node
    def branch_b(state: PState) -> dict[str, list[str]]:
        return {"results": ["b"]}

    @graph.node
    def join(state: PState) -> PState:
        state.value = len(state.results)
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a, branch_b, join=join)

    result = graph.compile().run(PState())

    assert result.status == RunStatus.COMPLETED
    assert result.state.value == 2
    assert sorted(result.state.results) == ["a", "b"]


def test_parallel_branches_run_concurrently() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    async def slow_a(state: PState) -> dict[str, list[str]]:
        await asyncio.sleep(0.2)
        return {"results": ["a"]}

    @graph.node
    async def slow_b(state: PState) -> dict[str, list[str]]:
        await asyncio.sleep(0.2)
        return {"results": ["b"]}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, slow_a, slow_b)

    started = time.perf_counter()
    result = graph.compile().run(PState())
    elapsed = time.perf_counter() - started

    assert result.status == RunStatus.COMPLETED
    assert sorted(result.state.results) == ["a", "b"]
    assert elapsed < 0.35, f"branches did not overlap (took {elapsed:.2f}s)"


def test_parallel_branch_mutates_its_own_copy() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def mutate_a(state: PState) -> PState:
        state.results.append("a")
        return state

    @graph.node
    def mutate_b(state: PState) -> PState:
        state.results.append("b")
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, mutate_a, mutate_b)

    result = graph.compile().run(PState())

    assert result.status == RunStatus.COMPLETED
    # in-place mutations of isolated copies are merged back through reducers
    assert sorted(result.state.results) == ["a", "b"]


def test_parallel_unreduced_field_is_last_writer_wins() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def set_one(state: PState) -> dict[str, int]:
        return {"value": 1}

    @graph.node
    def set_two(state: PState) -> dict[str, int]:
        return {"value": 2}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, set_one, set_two)

    result = graph.compile().run(PState())

    assert result.status == RunStatus.COMPLETED
    assert result.state.value in (1, 2)


def test_parallel_without_join_ends() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def branch(state: PState) -> dict[str, list[str]]:
        return {"results": ["x"]}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch, join=END)

    result = graph.compile().run(PState())

    assert result.status == RunStatus.COMPLETED
    assert result.state.results == ["x"]


def test_parallel_branch_failure_fails_run() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def ok_branch(state: PState) -> dict[str, list[str]]:
        return {"results": ["ok"]}

    @graph.node
    def boom_branch(state: PState) -> PState:
        raise RuntimeError("branch failed")

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, ok_branch, boom_branch)

    result = graph.compile().run(PState())

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "branch failed" in result.error


def test_parallel_requires_targets() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    graph.set_entry_point(start)
    with pytest.raises(GraphCompileError):
        graph.add_parallel_edges(start)


def test_parallel_conflicts_with_normal_edges() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def other(state: PState) -> PState:
        return state

    graph.set_entry_point(start)
    graph.add_edge(start, other)
    graph.add_parallel_edges(start, other)

    with pytest.raises(GraphCompileError):
        graph.compile()


def test_parallel_unknown_target_fails_compile() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, "ghost")  # type: ignore[arg-type]

    with pytest.raises(GraphCompileError):
        graph.compile()


def test_parallel_targets_reachable_from_entry() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def unreachable_branch(state: PState) -> PState:
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, unreachable_branch)

    # Should compile cleanly: parallel targets are reachable.
    assert graph.compile() is not None


def test_parallel_events_include_branch_nodes() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def branch_a(state: PState) -> dict[str, list[str]]:
        return {"results": ["a"]}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a)

    result = graph.compile().run(PState())
    nodes = {event.node for event in result.events}

    assert "branch_a" in nodes
