"""Run object and structured execution events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic

from pyantra.state.state import StateT


class RunStatus(str, Enum):
    """Lifecycle status of a workflow run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class RunEvent:
    """A structured execution event emitted during a run."""

    run_id: str
    event: str
    timestamp: float
    node: str | None = None
    duration_ms: float | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "event": self.event,
            "timestamp": self.timestamp,
            "node": self.node,
            "duration_ms": self.duration_ms,
            "message": self.message,
        }


@dataclass
class Run(Generic[StateT]):
    """The result of a single workflow execution.

    ``state`` holds the final (or last known) state, ``events`` contains the
    structured trace, and ``error`` describes the failure when ``status`` is
    ``FAILED``.
    """

    run_id: str
    status: RunStatus
    state: StateT | None = None
    events: list[RunEvent] = field(default_factory=list)
    error: str | None = None
    exception: BaseException | None = None

    @property
    def node_events(self) -> list[RunEvent]:
        """Events that reference a specific node."""
        return [event for event in self.events if event.node is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "state": self.state,
            "events": [event.to_dict() for event in self.events],
            "error": self.error,
        }
