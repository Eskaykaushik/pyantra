"""Node primitive: a single executable step in a workflow graph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias

from pyantra.reliability.circuit_breaker import CircuitBreaker
from pyantra.reliability.retry import Backoff
from pyantra.state.state import StateT, StateUpdate

NodeFn: TypeAlias = Callable[
    [StateT], StateT | StateUpdate | None | Awaitable[StateT | StateUpdate | None]
]


@dataclass(frozen=True)
class NodeConfig:
    """Reliability configuration for a node.

    ``retries`` is the number of retries after the first attempt. ``backoff``
    selects the delay strategy (fixed or exponential). ``timeout`` limits a
    single attempt in seconds. ``breaker`` guards the node against a run of
    failures across runs. ``retry_on`` restricts which failures are retried
    to the given exception type(s); anything else fails immediately. A single
    type is accepted as shorthand for a one-element tuple.
    """

    retries: int = 0
    backoff: Backoff = Backoff.FIXED
    base_delay: float = 1.0
    max_delay: float | None = None
    timeout: float | None = None
    breaker: CircuitBreaker | None = None
    retry_on: tuple[type[Exception], ...] | type[Exception] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.retry_on, type):
            object.__setattr__(self, "retry_on", (self.retry_on,))


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

    def __call__(
        self, state: StateT
    ) -> StateT | StateUpdate | None | Awaitable[StateT | StateUpdate | None]:
        return self.fn(state)

    def __repr__(self) -> str:
        return f"Node({self.name!r})"
