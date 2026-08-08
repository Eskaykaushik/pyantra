"""SQLite-backed checkpoint store."""

from __future__ import annotations

import pickle
import sqlite3
import threading
import time

from pyantra.checkpoint.base import Checkpoint, CheckpointStore
from pyantra.state.state import StateT

_CREATE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id     TEXT PRIMARY KEY,
    resume_at  TEXT,
    state      BLOB NOT NULL,
    events     BLOB NOT NULL,
    interrupts BLOB NOT NULL,
    updated_at REAL NOT NULL
)
"""


class SQLiteCheckpointStore(CheckpointStore[StateT]):
    """A durable checkpoint store backed by a single SQLite database.

    State, events, and interrupt payloads are serialized with ``pickle``, so
    any picklable state type works. Thread-safe; usable as a context manager
    to ensure the connection is closed.
    """

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE)
        self._conn.commit()

    def save(self, checkpoint: Checkpoint[StateT]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO checkpoints "
                "(run_id, resume_at, state, events, interrupts, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "resume_at=excluded.resume_at, state=excluded.state, "
                "events=excluded.events, interrupts=excluded.interrupts, "
                "updated_at=excluded.updated_at",
                (
                    checkpoint.run_id,
                    checkpoint.resume_at,
                    pickle.dumps(checkpoint.state),
                    pickle.dumps(checkpoint.events),
                    pickle.dumps(checkpoint.interrupts),
                    time.time(),
                ),
            )
            self._conn.commit()

    def load(self, run_id: str) -> Checkpoint[StateT] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT resume_at, state, events, interrupts "
                "FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            run_id=run_id,
            resume_at=row[0],
            state=pickle.loads(row[1]),
            events=pickle.loads(row[2]),
            interrupts=pickle.loads(row[3]),
        )

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteCheckpointStore[StateT]:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["SQLiteCheckpointStore"]
