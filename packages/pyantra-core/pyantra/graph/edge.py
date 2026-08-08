"""Edge primitive: defines unconditional execution order."""

from __future__ import annotations

from dataclasses import dataclass


class _End:
    """Sentinel marking the end of a workflow execution."""

    def __repr__(self) -> str:
        return "END"


END = _End()


@dataclass(frozen=True)
class Edge:
    """An unconditional edge from a source node to a target node.

    ``target`` is ``None`` when the edge terminates the workflow (``END``).
    """

    source: str
    target: str | None
