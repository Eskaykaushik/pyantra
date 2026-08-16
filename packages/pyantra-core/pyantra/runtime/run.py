"""Run object and structured execution events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic

from pyantra.llm.types import Usage
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
    usage: Usage | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "event": self.event,
            "timestamp": self.timestamp,
            "node": self.node,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "usage": _usage_to_dict(self.usage) if self.usage is not None else None,
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
    interrupt: Any = None
    usage: Usage = field(default_factory=Usage)

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
            "interrupt": self.interrupt,
            "usage": _usage_to_dict(self.usage),
        }


def _usage_to_dict(usage: Usage) -> dict[str, object]:
    """Serialize a :class:`~pyantra.llm.types.Usage` for JSON-friendly output."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_tokens": usage.cache_tokens,
        "cost": usage.cost,
        "model": usage.model,
    }
