"""Tests for field reducers and partial state updates."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated

from pyantra import Graph, NodeExecutionError, RunStatus
from pyantra.state.reducers import (
    apply_updates,
    diff_state,
    extract_reducers,
    merge_state,
)


@dataclass
class ReducibleState:
    value: int = 0
    messages: Annotated[list[str], operator.add] = field(default_factory=list)
    tags: Annotated[set[str], operator.or_] = field(default_factory=set)


def test_extract_reducers_from_annotated_fields() -> None:
    reducers = extract_reducers(ReducibleState)
    assert reducers["messages"] == operator.add
    assert reducers["tags"] == operator.or_
    assert "value" not in reducers


def test_extract_reducers_ignores_plain_types() -> None:
    @dataclass
    class Plain:
        value: int

    assert extract_reducers(Plain) == {}


def test_apply_updates_runs_reducers() -> None:
    state = ReducibleState(value=1, messages=["a"])
    reducers = extract_reducers(ReducibleState)
    updated = apply_updates(state, {"messages": ["b"], "tags": {"x"}}, reducers)
    assert updated is state
    assert state.messages == ["a", "b"]
    assert state.tags == {"x"}


def test_apply_updates_overwrites_plain_fields() -> None:
    state = ReducibleState(value=1)
    apply_updates(state, {"value": 42}, {})
    assert state.value == 42


def test_apply_updates_unknown_field_raises() -> None:
    state = ReducibleState()
    try:
        apply_updates(state, {"nope": 1}, {})
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_merge_state_reduces_returned_object() -> None:
    current = ReducibleState(messages=["a"])
    returned = ReducibleState(value=9, messages=["b"])
    reducers = extract_reducers(ReducibleState)
    merged = merge_state(current, returned, reducers)
    assert merged is current
    assert merged.value == 9
    assert merged.messages == ["a", "b"]


def test_merge_state_identity_means_in_place() -> None:
    current = ReducibleState(messages=["a"])
    reducers = extract_reducers(ReducibleState)
    merged = merge_state(current, current, reducers)
    assert merged.messages == ["a"]


def test_diff_state_strips_base_prefix_from_reducer_field() -> None:
    snapshot = ReducibleState(messages=["base"])
    returned = ReducibleState(messages=["base", "a"])
    reducers = extract_reducers(ReducibleState)
    assert diff_state(snapshot, returned, reducers) == {
        "value": 0,
        "messages": ["a"],
        "tags": set(),
    }


def test_diff_state_rebind_matches_prefix() -> None:
    snapshot = ReducibleState(messages=["base"])
    returned = ReducibleState(messages=["base", "a", "b"])
    reducers = extract_reducers(ReducibleState)
    assert diff_state(snapshot, returned, reducers)["messages"] == ["a", "b"]


def test_diff_state_non_prefix_value_is_whole_delta() -> None:
    snapshot = ReducibleState(messages=["base"])
    returned = ReducibleState(messages=["a"])
    reducers = extract_reducers(ReducibleState)
    assert diff_state(snapshot, returned, reducers)["messages"] == ["a"]


def test_diff_state_passes_non_reducer_fields_whole() -> None:
    snapshot = ReducibleState(value=1)
    returned = ReducibleState(value=9, messages=["a"])
    reducers = extract_reducers(ReducibleState)
    updates = diff_state(snapshot, returned, reducers)
    assert updates["value"] == 9
    assert updates["messages"] == ["a"]


def test_diff_state_delta_feeds_apply_updates() -> None:
    snapshot = ReducibleState(messages=["base"])
    returned = ReducibleState(messages=["base", "a"])
    reducers = extract_reducers(ReducibleState)
    merged = apply_updates(
        ReducibleState(messages=["base"]),
        diff_state(snapshot, returned, reducers),
        reducers,
    )
    assert merged.messages == ["base", "a"]


def test_node_partial_update_dict() -> None:
    graph = Graph(ReducibleState)

    @graph.node
    def append(state: ReducibleState) -> dict[str, list[str]]:
        return {"messages": ["hello"]}

    graph.set_entry_point(append)

    result = graph.compile().run(ReducibleState())

    assert result.status == RunStatus.COMPLETED
    assert result.state.messages == ["hello"]


def test_node_returned_object_merges_with_reducers() -> None:
    graph = Graph(ReducibleState)

    @graph.node
    def add_one(state: ReducibleState) -> ReducibleState:
        return ReducibleState(value=state.value + 1, messages=["step"])

    graph.set_entry_point(add_one)

    result = graph.compile().run(ReducibleState(messages=["seed"]))

    assert result.state.value == 1
    assert result.state.messages == ["seed", "step"]


def test_node_in_place_mutation_still_works() -> None:
    graph = Graph(ReducibleState)

    @graph.node
    def mutate(state: ReducibleState) -> ReducibleState:
        state.messages.append("in-place")
        return state

    graph.set_entry_point(mutate)

    result = graph.compile().run(ReducibleState(messages=["seed"]))

    assert result.state.messages == ["seed", "in-place"]


def test_reducer_runs_through_pipeline() -> None:
    graph = Graph(ReducibleState)

    @graph.node
    def step1(state: ReducibleState) -> dict[str, list[str]]:
        return {"messages": ["one"]}

    @graph.node
    def step2(state: ReducibleState) -> dict[str, list[str]]:
        return {"messages": ["two"]}

    graph.set_entry_point(step1)
    graph.add_edge(step1, step2)

    result = graph.compile().run(ReducibleState())

    assert result.state.messages == ["one", "two"]


def test_unknown_partial_update_field_fails_run() -> None:
    graph = Graph(ReducibleState)

    @graph.node
    def bad(state: ReducibleState) -> dict[str, object]:
        return {"bogus": 1}

    graph.set_entry_point(bad)

    result = graph.compile().run(ReducibleState())

    assert result.status == RunStatus.FAILED
    assert isinstance(result.exception, NodeExecutionError)
    assert "bogus" in result.error or "merge" in result.error
