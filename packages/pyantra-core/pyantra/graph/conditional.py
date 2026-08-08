"""Conditional routing primitive."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Generic, TypeAlias

from pyantra.state.state import StateT

RouterFn: TypeAlias = Callable[[StateT], str | Awaitable[str]]


class ConditionalEdge(Generic[StateT]):
    """Routes execution based on a router function's output.

    Two forms are supported:

    * ``path_map`` given — the router returns a key into ``path_map``, which
      maps keys to node names. An optional ``default`` is used when the router
      returns an unknown key.
    * ``path_map`` omitted — the router returns a registered node name directly.
    """

    def __init__(
        self,
        source: str,
        router: RouterFn[StateT],
        path_map: Mapping[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self.source = source
        self.router = router
        self.path_map = path_map
        self.default = default

    def __repr__(self) -> str:
        return f"ConditionalEdge(source={self.source!r})"
