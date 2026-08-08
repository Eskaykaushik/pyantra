# Checkpoints and Resume

A **checkpoint** is a durable snapshot of a run — its state, event trace, and
pending interrupts. Pass a checkpoint store to `run()` and a failed run can
resume from its last successful node instead of restarting.

## The checkpoint store interface

`CheckpointStore` is an abstract interface with three operations:

```python
class CheckpointStore(ABC):
    def save(self, checkpoint: Checkpoint) -> None: ...
    def load(self, run_id: str) -> Checkpoint | None: ...
    def delete(self, run_id: str) -> None: ...
```

Two implementations ship with Pyantra:

- `MemoryCheckpointStore` — an in-process dictionary (not durable).
- `SQLiteCheckpointStore` — a single-file SQLite database (durable).

```python
from pyantra import MemoryCheckpointStore, SQLiteCheckpointStore

memory = MemoryCheckpointStore()
sqlite = SQLiteCheckpointStore("checkpoints.db")
```

`SQLiteCheckpointStore` serializes state, events, and interrupt payloads with
`pickle`, so any picklable state type works and the database survives process
restarts. It is thread-safe and usable as a context manager:

```python
with SQLiteCheckpointStore("checkpoints.db") as store:
    ...
```

## Resuming a failed run

Pass `checkpointer` and a stable `run_id` to `run()`. If the run fails partway,
re-running with the same `run_id` resumes from the last successful node:

```python
store = MemoryCheckpointStore()

first = app.run(state, checkpointer=store, run_id="order-123")
assert first.status == RunStatus.FAILED

# Same run_id: resumes where it stopped instead of restarting.
second = app.run(state, checkpointer=store, run_id="order-123")
```

A run is checkpointed after each node completes. On failure, `run_id` is
reused; on success, the checkpoint is deleted automatically.

## What a checkpoint holds

```python
from pyantra import Checkpoint

# Checkpoint fields:
#   run_id       — the run this snapshot belongs to
#   resume_at    — the node to (re-)execute on resume
#   state        — the state at the last successful node
#   events       — the event trace so far
#   interrupts   — pending human-in-the-loop (node, payload) pairs
```

## Custom stores

Implement the interface for other backends (Postgres, Redis, …). Stores are
**pure storage** — they do not interpret checkpoint contents, so a backend is
just serialization:

```python
from pyantra import Checkpoint, CheckpointStore

class MyStore(CheckpointStore[State]):
    def save(self, checkpoint: Checkpoint[State]) -> None:
        ...

    def load(self, run_id: str) -> Checkpoint[State] | None:
        ...

    def delete(self, run_id: str) -> None:
        ...
```

Storage failures raise `CheckpointError` — see [Errors](errors.md).

## Related guides

- [Human-in-the-loop](human-in-the-loop.md) — `interrupt()` uses the same
  checkpointer to pause and resume runs.
- [Observability](observability.md) — the event trace stored in checkpoints.
