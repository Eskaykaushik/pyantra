"""Per-node circuit breaker.

A circuit breaker stops execution of a node after a run of consecutive
failures, then allows a trial call once a reset period has elapsed. Breaker
state lives on the node and is shared across runs.
"""

from __future__ import annotations

import time
from enum import Enum

from pyantra.runtime.errors import CircuitOpenError


class CircuitState(str, Enum):
    """Lifecycle state of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Tracks node failures and refuses execution while open.

    * ``CLOSED`` — normal operation. Consecutive failures accumulate.
    * ``OPEN`` — after ``failure_threshold`` consecutive failures, execution
      is refused until ``reset_timeout`` elapses.
    * ``HALF_OPEN`` — after the reset period, one trial call is allowed. A
      success closes the circuit; a failure reopens it.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        *,
        name: str | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if time.monotonic() - self._opened_at < self.reset_timeout:
            return CircuitState.OPEN
        return CircuitState.HALF_OPEN

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    def before(self, *, node: str | None = None, run_id: str | None = None) -> None:
        """Refuse execution with ``CircuitOpenError`` when the circuit is open."""
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit is open for node {node!r}; refusing execution.",
                run_id=run_id,
                node=node,
            )

    def record_success(self) -> bool:
        """Record a successful execution. Returns True if the circuit closed."""
        was_open = self._opened_at is not None
        self._failures = 0
        self._opened_at = None
        return was_open

    def record_failure(self) -> bool:
        """Record a failed execution. Returns True if the circuit opened."""
        if self.state == CircuitState.HALF_OPEN:
            self._opened_at = time.monotonic()
            self._failures = 0
            return True
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
            self._failures = 0
            return True
        return False

    def reset(self) -> None:
        """Manually close the circuit and clear failure counts."""
        self._failures = 0
        self._opened_at = None
