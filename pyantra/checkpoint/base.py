"""Checkpoint storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic

from pyantra.runtime.run import RunEvent
from pyantra.state.state import StateT


@dataclass
class Checkpoint(Generic[StateT]):
    """A snapshot of a run, used to resume after a failure.

    ``resume_at`` is the name of the node to (re-)execute on resume — the node
    that was executing when the run failed.
    """

    run_id: str
    resume_at: str | None
    state: StateT
    events: list[RunEvent] = field(default_factory=list)


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
