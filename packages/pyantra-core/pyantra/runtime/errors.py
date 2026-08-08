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


class NodeTimeoutError(GraphExecutionError):
    """Raised when a node exceeds its configured timeout."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        node: str | None = None,
    ) -> None:
        super().__init__(message, run_id=run_id)
        self.node = node


class RetryExhaustedError(GraphExecutionError):
    """Raised when a node fails after exhausting its retry policy."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        node: str | None = None,
    ) -> None:
        super().__init__(message, run_id=run_id)
        self.node = node


class CircuitOpenError(GraphExecutionError):
    """Raised when a node's circuit breaker is open and refuses execution."""

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


class CheckpointError(PyantraError):
    """Raised for checkpoint storage or resume failures."""


class NonRetryableError(PyantraError):
    """Base class for errors that must never be retried."""
