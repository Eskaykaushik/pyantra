"""Parallel fan-out primitive."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelEdge:
    """Fan out from a source node to several targets that run concurrently.

    Each target executes on an isolated copy of the current state and returns
    its result. Results are merged back into the shared state using the field
    reducers (unannotated fields are last-writer-wins). Execution then
    continues at ``join``, or terminates when ``join`` is ``None``.
    """

    source: str
    targets: tuple[str, ...]
    join: str | None


__all__ = ["ParallelEdge"]
