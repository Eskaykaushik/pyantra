"""Node primitive: a single executable step in a workflow graph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Generic, TypeAlias

from pyantra.state.state import StateT

NodeFn: TypeAlias = Callable[[StateT], StateT | None | Awaitable[StateT | None]]


class Node(Generic[StateT]):
    """A single executable step in a workflow graph.

    A node receives state and returns an updated state. Returning ``None``
    signals that the node mutated state in place.
    """

    def __init__(self, name: str, fn: NodeFn[StateT]) -> None:
        self.name = name
        self.fn = fn

    def __call__(self, state: StateT) -> StateT | None | Awaitable[StateT | None]:
        return self.fn(state)

    def __repr__(self) -> str:
        return f"Node({self.name!r})"
