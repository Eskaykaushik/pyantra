"""Tests for conditional routing."""

from __future__ import annotations

from conftest import State

from pyantra import Graph, RunStatus


def _build_positive_negative_graph() -> Graph[State]:
    graph = Graph(State)

    @graph.node
    def classify(state: State) -> State:
        state.history.append("classify")
        return state

    @graph.node
    def process_positive(state: State) -> State:
        state.history.append("positive")
        return state

    @graph.node
    def process_negative(state: State) -> State:
        state.history.append("negative")
        return state

    def route(state: State) -> str:
        return "positive" if state.value >= 0 else "negative"

    graph.set_entry_point(classify)
    graph.add_conditional_edges(
        classify,
        route,
        {"positive": process_positive, "negative": process_negative},
    )
    return graph


def test_conditional_positive_path() -> None:
    result = _build_positive_negative_graph().compile().run(State(value=1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.history == ["classify", "positive"]


def test_conditional_negative_path() -> None:
    result = _build_positive_negative_graph().compile().run(State(value=-1))

    assert result.status == RunStatus.COMPLETED
    assert result.state is not None
    assert result.state.history == ["classify", "negative"]


async def test_conditional_async() -> None:
    result = await _build_positive_negative_graph().compile().arun(State(value=-5))

    assert result.state is not None
    assert result.state.history == ["classify", "negative"]


async def test_conditional_with_async_router() -> None:
    graph = Graph(State)

    @graph.node
    def classify(state: State) -> State:
        state.history.append("classify")
        return state

    @graph.node
    def big(state: State) -> State:
        state.history.append("big")
        return state

    @graph.node
    def small(state: State) -> State:
        state.history.append("small")
        return state

    async def route(state: State) -> str:
        return "big" if state.value > 5 else "small"

    graph.set_entry_point(classify)
    graph.add_conditional_edges(classify, route, {"big": big, "small": small})

    result = await graph.compile().arun(State(value=10))

    assert result.state is not None
    assert result.state.history == ["classify", "big"]


def test_router_returning_node_name_directly() -> None:
    graph = Graph(State)

    @graph.node
    def classify(state: State) -> State:
        state.history.append("classify")
        return state

    @graph.node
    def flagged(state: State) -> State:
        state.history.append("flagged")
        return state

    @graph.node
    def unflagged(state: State) -> State:
        state.history.append("unflagged")
        return state

    def route(state: State) -> str:
        return "flagged" if state.value == 0 else "unflagged"

    graph.set_entry_point(classify)
    graph.add_conditional_edges(classify, route)

    result = graph.compile().run(State(value=0))

    assert result.state is not None
    assert result.state.history == ["classify", "flagged"]


def test_default_route_used_for_unknown_key() -> None:
    graph = Graph(State)

    @graph.node
    def classify(state: State) -> State:
        return state

    @graph.node
    def known(state: State) -> State:
        state.history.append("known")
        return state

    @graph.node
    def fallback(state: State) -> State:
        state.history.append("fallback")
        return state

    def route(state: State) -> str:
        return "unexpected"

    graph.set_entry_point(classify)
    graph.add_conditional_edges(
        classify,
        route,
        {"known": known},
        default=fallback,
    )

    result = graph.compile().run(State())

    assert result.state is not None
    assert result.state.history == ["fallback"]


def test_unknown_route_fails_run() -> None:
    graph = Graph(State)

    @graph.node
    def classify(state: State) -> State:
        return state

    @graph.node
    def known(state: State) -> State:
        return state

    def route(state: State) -> str:
        return "unexpected"

    graph.set_entry_point(classify)
    graph.add_conditional_edges(classify, route, {"known": known})

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "unexpected" in result.error


def test_router_returning_unknown_node_fails_run() -> None:
    graph = Graph(State)

    @graph.node
    def classify(state: State) -> State:
        return state

    @graph.node
    def known(state: State) -> State:
        return state

    def route(state: State) -> str:
        return "ghost"

    graph.set_entry_point(classify)
    graph.add_conditional_edges(classify, route)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "ghost" in result.error
