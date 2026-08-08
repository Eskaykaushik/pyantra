"""In-memory checkpoint store."""

from __future__ import annotations

from pyantra.checkpoint.base import Checkpoint, CheckpointStore
from pyantra.state.state import StateT


class MemoryCheckpointStore(CheckpointStore[StateT]):
    """A checkpoint store backed by an in-process dictionary."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint[StateT]] = {}

    def save(self, checkpoint: Checkpoint[StateT]) -> None:
        self._store[checkpoint.run_id] = checkpoint

    def load(self, run_id: str) -> Checkpoint[StateT] | None:
        return self._store.get(run_id)

    def delete(self, run_id: str) -> None:
        self._store.pop(run_id, None)
