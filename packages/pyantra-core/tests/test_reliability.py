"""Tests for retry, timeout, and circuit breaker reliability features."""

from __future__ import annotations

import asyncio
import time

from conftest import State

from pyantra import (
    Backoff,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    Graph,
    NodeConfig,
    NodeExecutionError,
    NodeTimeoutError,
    RetryExhaustedError,
    RunStatus,
    compute_delay,
    is_retryable,
    non_retryable,
)


def test_compute_delay_fixed() -> None:
    assert compute_delay(Backoff.FIXED, 1, 2.0) == 2.0
    assert compute_delay(Backoff.FIXED, 3, 2.0) == 2.0
    assert compute_delay(Backoff.FIXED, 2, 2.0, max_delay=1.0) == 1.0


def test_compute_delay_exponential() -> None:
    assert compute_delay(Backoff.EXPONENTIAL, 1, 1.0) == 1.0
    assert compute_delay(Backoff.EXPONENTIAL, 2, 1.0) == 2.0
    assert compute_delay(Backoff.EXPONENTIAL, 3, 1.0) == 4.0
    assert compute_delay(Backoff.EXPONENTIAL, 5, 1.0, max_delay=3.0) == 3.0


def test_compute_delay_none_or_zero() -> None:
    assert compute_delay(Backoff.NONE, 2, 1.0) == 0.0
    assert compute_delay(Backoff.FIXED, 2, 0.0) == 0.0


def test_is_retryable_default_true() -> None:
    assert is_retryable(RuntimeError("x"))


@non_retryable
class _BadInputError(Exception):
    pass


def test_non_retryable_marker() -> None:
    assert not is_retryable(_BadInputError("bad"))


def test_retry_succeeds_after_failures() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        state.history.append("done")
        return state

    n.config = NodeConfig(retries=3, backoff=Backoff.NONE)

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.COMPLETED
    assert calls["n"] == 3
    assert result.state is not None
    assert result.state.history == ["done"]
    assert "node.retrying" in {e.event for e in result.events}


def test_retry_exhausted_raises_retry_exhausted_error() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        raise ValueError("always fails")

    n.config = NodeConfig(retries=2, backoff=Backoff.NONE)

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert calls["n"] == 3
    assert isinstance(result.exception, RetryExhaustedError)
    assert result.exception.node == "n"
    assert isinstance(result.exception.__cause__, ValueError)


def test_non_retryable_error_skips_retries() -> None:
    graph = Graph(State)
    calls = {"n": 0}

    @non_retryable
    class _Permanent(Exception):
        pass

    @graph.node
    def n(state: State) -> State:
        calls["n"] += 1
        raise _Permanent("no point retrying")

    n.config = NodeConfig(retries=3, backoff=Backoff.NONE)

    graph.set_entry_point(n)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert calls["n"] == 1
    assert isinstance(result.exception, NodeExecutionError)


def test_timeout_raises_node_timeout_error() -> None:
    graph = Graph(State)

    async def slow(state: State) -> State:
        await asyncio.sleep(10)
        return state

    slow_node = graph.node(slow, config=NodeConfig(timeout=0.05))
    graph.set_entry_point(slow_node)

    result = graph.compile().run(State())

    assert result.status == RunStatus.FAILED
    assert isinstance(result.exception, NodeTimeoutError)
    assert result.exception.node == "slow"


def test_circuit_breaker_state_transitions() -> None:
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN  # type: ignore[comparison-overlap]
    assert breaker.record_failure() is False

    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_circuit_breaker_blocks_execution_after_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
    graph = Graph(State)
    calls = {"n": 0}

    def flaky(state: State) -> State:
        calls["n"] += 1
        raise RuntimeError("boom")

    graph.add_node(flaky, name="n", config=NodeConfig(breaker=breaker))
    graph.set_entry_point("n")

    app = graph.compile()

    first = app.run(State())
    assert first.status == RunStatus.FAILED
    assert isinstance(first.exception, NodeExecutionError)

    second = app.run(State())
    assert second.status == RunStatus.FAILED

    third = app.run(State())
    assert third.status == RunStatus.FAILED
    assert isinstance(third.exception, CircuitOpenError)
    assert third.exception.node == "n"
    assert breaker.state == CircuitState.OPEN
    assert calls["n"] == 2


def test_circuit_breaker_half_open_trial_and_reopen() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
    graph = Graph(State)
    calls = {"n": 0}

    def flaky(state: State) -> State:
        calls["n"] += 1
        raise RuntimeError("boom")

    graph.add_node(flaky, name="n", config=NodeConfig(breaker=breaker))
    graph.set_entry_point("n")

    app = graph.compile()
    app.run(State())
    assert breaker.state == CircuitState.OPEN
    assert calls["n"] == 1

    time.sleep(0.1)
    assert breaker.state == CircuitState.HALF_OPEN  # type: ignore[comparison-overlap]

    trial = app.run(State())
    assert isinstance(trial.exception, NodeExecutionError)
    assert calls["n"] == 2
    assert breaker.state == CircuitState.OPEN
