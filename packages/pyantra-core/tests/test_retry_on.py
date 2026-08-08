"""Tests for the ``retry_on`` retry filter."""

from __future__ import annotations

import asyncio

from conftest import State

from pyantra import (
    Backoff,
    Graph,
    NodeConfig,
    NodeExecutionError,
    NodeTimeoutError,
    RunStatus,
    non_retryable,
)


class _TransientError(Exception):
    pass


class _PermanentError(Exception):
    pass


def test_retry_on_matching_exception_retries_until_success() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _TransientError("transient")
        state.history.append("done")
        return state

    n.config = NodeConfig(retries=3, backoff=Backoff.NONE, retry_on=(_TransientError,))

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.COMPLETED
    assert calls["n"] == 3
    assert result.state is not None
    assert result.state.history == ["done"]


def test_retry_on_excludes_non_matching_exception() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        raise _PermanentError("never worth retrying")

    n.config = NodeConfig(retries=3, backoff=Backoff.NONE, retry_on=(_TransientError,))

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert calls["n"] == 1
    assert isinstance(result.exception, NodeExecutionError)
    assert isinstance(result.exception.__cause__, _PermanentError)


def test_retry_on_exhaustion_raises_retry_exhausted_error() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        raise _TransientError("always transient")

    n.config = NodeConfig(retries=2, backoff=Backoff.NONE, retry_on=(_TransientError,))

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert calls["n"] == 3
    assert result.exception is not None
    assert result.exception.node == "n"


def test_retry_on_accepts_single_type_shorthand() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _TransientError("one retry needed")
        state.history.append("done")
        return state

    n.config = NodeConfig(retries=2, backoff=Backoff.NONE, retry_on=_TransientError)

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.COMPLETED
    assert calls["n"] == 2


def test_retry_on_with_subclass_matches() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    class _SpecificError(_TransientError):
        pass

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _SpecificError("subclass of transient")
        state.history.append("done")
        return state

    n.config = NodeConfig(retries=2, backoff=Backoff.NONE, retry_on=(_TransientError,))

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.COMPLETED
    assert calls["n"] == 2


def test_non_retryable_wins_over_retry_on() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @non_retryable
    class _NeverRetry(Exception):
        pass

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        raise _NeverRetry("marked non-retryable")

    n.config = NodeConfig(retries=3, backoff=Backoff.NONE, retry_on=(_NeverRetry,))

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert calls["n"] == 1
    assert isinstance(result.exception, NodeExecutionError)


def test_retry_on_filters_timeouts() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    async def n(state: State) -> State:
        calls["n"] += 1
        await asyncio.sleep(10)
        return state

    n.config = NodeConfig(
        retries=3,
        backoff=Backoff.NONE,
        timeout=0.05,
        retry_on=(_TransientError,),
    )

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert calls["n"] == 1
    assert isinstance(result.exception, NodeTimeoutError)


def test_retry_on_includes_timeout_retries_timeout() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    async def n(state: State) -> State:
        calls["n"] += 1
        if calls["n"] < 3:
            await asyncio.sleep(10)
        state.history.append("done")
        return state

    n.config = NodeConfig(
        retries=3,
        backoff=Backoff.NONE,
        timeout=0.05,
        retry_on=(TimeoutError,),
    )

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.COMPLETED
    assert calls["n"] == 3
    assert result.state is not None
    assert result.state.history == ["done"]
