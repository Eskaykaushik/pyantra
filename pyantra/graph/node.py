"""Node primitive: a single executable step in a workflow graph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias

from pyantra.reliability.circuit_breaker import CircuitBreaker
from pyantra.reliability.retry import Backoff
from pyantra.state.state import StateT

NodeFn: TypeAlias = Callable[
    [StateT], StateT | None | Awaitable[StateT | None]
]


@dataclass(frozen=True)
class NodeConfig:
    """Reliability configuration for a node.

    ``retries`` is the number of retries after the first attempt. ``backoff``
    selects the delay strategy (fixed or exponential). ``timeout`` limits a
    single attempt in seconds. ``breaker`` guards the node against a run of
    failures across runs.
    """

    retries: int = 0
    backoff: Backoff = Backoff.FIXED
    base_delay: float = 1.0
    max_delay: float | None = None
    timeout: float | None = None
    breaker: CircuitBreaker | None = None


class Node(Generic[StateT]):
    """A single executable step in a workflow graph.

    A node receives state and returns an updated state. Returning ``None``
    signals that the node mutated state in place.
    """

    def __init__(
        self,
        name: str,
        fn: NodeFn[StateT],
        config: NodeConfig | None = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.config = config or NodeConfig()

    def __call__(self, state: StateT) -> StateT | None | Awaitable[StateT | None]:
        return self.fn(state)

    def __repr__(self) -> str:
        return f"Node({self.name!r})"
