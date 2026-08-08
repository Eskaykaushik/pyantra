"""Example: retry, timeout, and circuit breaker reliability features.

Run with:

    python examples/reliability_workflow.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pyantra import (
    Backoff,
    CircuitBreaker,
    CircuitOpenError,
    Graph,
    NodeConfig,
    NodeExecutionError,
    RetryExhaustedError,
    RunStatus,
)


@dataclass
class State:
    value: int
    attempts: list[str] = field(default_factory=list)


def retry_workflow() -> int:
    graph = Graph(State)
    calls = {"fetch": 0}

    @graph.node
    def fetch(state: State) -> State:
        calls["fetch"] += 1
        if calls["fetch"] < 3:
            raise RuntimeError("transient network error")
        state.value = 42
        state.attempts.append("fetch")
        return state

    fetch.config = NodeConfig(retries=4, backoff=Backoff.EXPONENTIAL, base_delay=0.01)

    graph.set_entry_point(fetch)

    result = graph.compile().run(State(value=0))

    assert result.status == RunStatus.COMPLETED
    assert calls["fetch"] == 3
    final = result.state.value if result.state else None
    print(f"retry: value={final}, calls={calls['fetch']}")
    return result.state.value if result.state else 0


def timeout_workflow() -> str:
    graph = Graph(State)

    @graph.node
    async def slow_fetch(state: State) -> State:
        await asyncio.sleep(10)
        return state

    slow_fetch.config = NodeConfig(timeout=0.05)

    graph.set_entry_point(slow_fetch)

    result = graph.compile().run(State(value=0))

    assert result.status == RunStatus.FAILED
    print(f"timeout: status={result.status.value}, error={result.error}")
    return str(result.status.value)


def circuit_breaker_workflow() -> str:
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
    graph = Graph(State)
    calls = {"external": 0}

    def external_api(state: State) -> State:
        calls["external"] += 1
        raise RuntimeError("api down")

    graph.add_node(external_api, name="external", config=NodeConfig(breaker=breaker))
    graph.set_entry_point("external")

    app = graph.compile()

    app.run(State(value=0))
    app.run(State(value=0))
    third = app.run(State(value=0))

    assert isinstance(third.exception, CircuitOpenError)
    print(
        f"circuit: status={third.status.value}, calls={calls['external']}, "
        f"breaker_state={breaker.state.value}"
    )
    return f"{calls['external']}/{breaker.state.value}"


def retry_exhausted_workflow() -> str:
    graph = Graph(State)

    @graph.node
    def always_fails(state: State) -> State:
        raise ValueError("will not recover")

    always_fails.config = NodeConfig(retries=2, backoff=Backoff.NONE)

    graph.set_entry_point(always_fails)

    result = graph.compile().run(State(value=0))

    assert isinstance(result.exception, RetryExhaustedError)
    print(
        f"exhausted: status={result.status.value}, "
        f"error={result.exception.node} failed after retries"
    )
    return str(result.status.value)


def retry_on_workflow() -> str:
    graph = Graph(State)
    calls = {"fetch": 0}

    class TransientError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    @graph.node
    def fetch(state: State) -> State:
        calls["fetch"] += 1
        if state.value == 0:
            raise BadRequestError("never retry this")
        raise TransientError("network blip")

    fetch.config = NodeConfig(
        retries=3, backoff=Backoff.NONE, retry_on=(TransientError,)
    )

    graph.set_entry_point(fetch)

    graph.compile().run(State(value=1))
    retried = calls["fetch"]

    bad = graph.compile().run(State(value=0))
    assert isinstance(bad.exception, NodeExecutionError)
    print(
        f"retry_on: transient calls={retried}, "
        f"bad_request calls={calls['fetch'] - retried}"
    )
    return f"{retried}/{calls['fetch'] - retried}"


def main() -> None:
    assert retry_workflow() == 42
    assert timeout_workflow() == RunStatus.FAILED.value
    assert circuit_breaker_workflow() == "2/open"
    assert retry_exhausted_workflow() == RunStatus.FAILED.value
    assert retry_on_workflow() == "4/1"
    print("All examples passed.")


if __name__ == "__main__":
    main()
