"""Tests for parallel fan-out and join execution."""

from __future__ import annotations

import asyncio
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated

import pytest

from pyantra import END, Graph, GraphCompileError, RunStatus
from pyantra.state.reducers import merge_dicts


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


def test_parallel_in_place_copy_preserves_base_reducer_state() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def branch_a(state: PState) -> PState:
        state.results.append("a")
        return state

    @graph.node
    def branch_b(state: PState) -> PState:
        state.results.append("b")
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a, branch_b)

    result = graph.compile().run(PState(results=["base"]))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    # pre-existing reducer state must not be re-appended once per branch
    assert result.state.results == ["base", "a", "b"]


def test_parallel_branch_can_read_base_reducer_state() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def branch_a(state: PState) -> PState:
        if "a" not in state.results:
            state.results.append("a")
        return state

    @graph.node
    def branch_b(state: PState) -> PState:
        state.results.append("b")
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a, branch_b)

    result = graph.compile().run(PState(results=["base"]))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    # the branch saw the base content and still contributed only its delta
    assert result.state.results == ["base", "a", "b"]


def test_parallel_fresh_state_object_merges_as_delta() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def branch_a(state: PState) -> PState:
        return PState(results=["a"])

    @graph.node
    def branch_b(state: PState) -> PState:
        return PState(results=["b"])

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a, branch_b)

    result = graph.compile().run(PState(results=["base"]))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.results == ["base", "a", "b"]


def test_parallel_dict_reducer_with_base_state() -> None:
    @dataclass
    class DictState:
        meta: Annotated[dict[str, str], merge_dicts] = field(default_factory=dict)

    graph = Graph(DictState)

    @graph.node
    def start(state: DictState) -> DictState:
        return state

    @graph.node
    def branch_a(state: DictState) -> DictState:
        state.meta["a"] = "1"
        return state

    @graph.node
    def branch_b(state: DictState) -> DictState:
        state.meta["b"] = "2"
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a, branch_b)

    result = graph.compile().run(DictState(meta={"base": "0"}))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.meta == {"base": "0", "a": "1", "b": "2"}


def test_parallel_set_reducer_with_base_state() -> None:
    @dataclass
    class SetState:
        tags: Annotated[set[str], operator.or_] = field(default_factory=set)

    graph = Graph(SetState)

    @graph.node
    def start(state: SetState) -> SetState:
        return state

    @graph.node
    def branch_a(state: SetState) -> SetState:
        state.tags.add("a")
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a)

    result = graph.compile().run(SetState(tags={"base"}))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.tags == {"base", "a"}


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


def test_parallel_duplicate_target_fails_compile() -> None:
    graph = Graph(PState)

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    def leaf(state: PState) -> dict[str, list[str]]:
        return {"results": ["x"]}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, leaf, leaf)

    with pytest.raises(GraphCompileError, match="duplicate"):
        graph.compile()


async def test_parallel_failure_cancels_sibling_tasks() -> None:
    graph = Graph(PState)
    marker = {"slow": False}

    @graph.node
    def start(state: PState) -> PState:
        return state

    @graph.node
    async def slow(state: PState) -> dict[str, list[str]]:
        await asyncio.sleep(0.3)
        marker["slow"] = True
        return {"results": ["slow"]}

    @graph.node
    def boom(state: PState) -> PState:
        raise RuntimeError("branch failed")

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, slow, boom)

    run = await graph.compile().arun(PState())

    assert run.status == RunStatus.FAILED
    assert not marker["slow"], "sibling kept running after the run failed"
    event_names = [e.event for e in run.events]
    assert event_names[-1] == "run.failed"
    assert "node.completed" not in event_names[
        event_names.index("run.failed") + 1 :
    ]
