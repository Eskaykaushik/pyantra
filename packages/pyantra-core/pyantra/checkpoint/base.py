"""Checkpoint storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic

from pyantra.runtime.run import RunEvent
from pyantra.state.state import StateT


@dataclass(frozen=True)
class ParallelProgress:
    """A fan-out that was in progress when a run paused.

    Captured when a ``GraphInterrupt`` is raised inside a parallel branch so a
    later ``resume()`` can re-enter the fan-out instead of replaying it. Branch
    names in ``completed`` already had their results merged into the
    checkpointed state; ``pending`` are the branches that still need to run
    (the interrupted branch plus any in-flight siblings that were cancelled),
    and ``interrupted`` is the branch that requested input.
    """

    source: str
    targets: tuple[str, ...]
    join: str | None
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    interrupted: str | None


@dataclass
class Checkpoint(Generic[StateT]):
    """A snapshot of a run, used to resume after a failure or interrupt.

    ``resume_at`` is the name of the node to (re-)execute on resume — the node
    that was executing when the run failed or paused. ``interrupts`` holds
    ``(node, payload)`` pairs for pending human-in-the-loop interruptions.
    ``parallel`` records an in-progress fan-out when the run paused inside a
    parallel branch, so resume can skip already-completed branches.
    """

    run_id: str
    resume_at: str | None
    state: StateT
    events: list[RunEvent] = field(default_factory=list)
    interrupts: list[tuple[str, Any]] = field(default_factory=list)
    parallel: ParallelProgress | None = None


class CheckpointStore(ABC, Generic[StateT]):
    """Abstract storage for run checkpoints.

    Implementations are pure storage; they do not interpret checkpoint
    contents. External backends (SQLite, Postgres, Redis) can be added by
    implementing this interface.
    """

    @abstractmethod
    def save(self, checkpoint: Checkpoint[StateT]) -> None:
        """Persist a checkpoint, replacing any existing checkpoint for its run."""

    @abstractmethod
    def load(self, run_id: str) -> Checkpoint[StateT] | None:
        """Load the latest checkpoint for ``run_id``, or None."""

    @abstractmethod
    def delete(self, run_id: str) -> None:
        """Delete all checkpoints for ``run_id``."""
