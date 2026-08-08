"""Pyantra runtime errors."""

from __future__ import annotations


class PyantraError(Exception):
    """Base class for all Pyantra errors."""


class GraphCompileError(PyantraError):
    """Raised when a graph fails compilation and validation."""


class GraphExecutionError(PyantraError):
    """Raised when a compiled graph fails during execution."""

    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id


class NodeExecutionError(GraphExecutionError):
    """Raised when a node raises during execution."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        node: str | None = None,
    ) -> None:
        super().__init__(message, run_id=run_id)
        self.node = node


class InvalidRouteError(GraphExecutionError):
    """Raised when a router returns an unknown destination."""


class MaxIterationsError(GraphExecutionError):
    """Raised when a run exceeds the configured iteration limit."""
