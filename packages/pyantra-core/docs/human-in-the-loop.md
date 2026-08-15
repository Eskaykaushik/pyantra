# Human-in-the-loop

Call `interrupt()` from inside a node to **pause a run and request input** from
a human. The run stops with `RunStatus.PAUSED`, its payload lands on
`run.interrupt`, and the state is checkpointed. Resume later with `resume()` —
the call to `interrupt()` then returns the value you provided.

```python
from pyantra import interrupt

@graph.node
def review(state: State) -> State:
    decision = interrupt({"question": "approve this change?", "draft": state.draft})
    state.decision = decision
    return state
```

Interrupts require a checkpointer so the run can be resumed later:

```python
store = MemoryCheckpointStore()

run = app.run(state, checkpointer=store, run_id="review-7")
assert run.status == RunStatus.PAUSED
print(run.interrupt)   # {"question": "...", "draft": "..."}

resumed = app.resume("review-7", "approved", checkpointer=store)
assert resumed.status == RunStatus.COMPLETED
assert resumed.state.decision == "approved"
```

## How it works

1. A node calls `interrupt(payload)`.
2. The run saves a checkpoint and returns with `RunStatus.PAUSED`; the payload
   is available on `run.interrupt`.
3. The caller inspects the payload, asks a human (or an approval flow), then
   calls `app.resume(run_id, value, checkpointer=store)`.
4. Execution continues from the interrupted node. The `interrupt()` call
   returns `value`, and the node proceeds as if nothing happened.

Async callers use `aresume()`:

```python
resumed = await app.aresume("review-7", "approved", checkpointer=store)
```

Sync callers that find themselves inside a running event loop (a FastAPI
handler, for example) can use `resume_sync()` instead — it blocks until the
resumed run finishes and works from both sync and async context:

```python
resumed = app.resume_sync("review-7", "approved", checkpointer=store)
```

## Multiple interrupts

A run may pause more than once. Each `resume()` answers the **most recent**
interruption; subsequent `interrupt()` calls pause again and appear on the next
`Run.interrupt`.

## Design notes

- `interrupt()` raises a signal that derives from `BaseException`, so a node's
  own `except Exception` cannot swallow it.
- Calling `interrupt()` without a checkpointer raises `PyantraError` — pass
  `checkpointer=...` to `run()`.
- The payload must be serializable by the checkpointer (JSON-serializable
  with the default serializer; any picklable value with `PickleSerializer`).

## Approval flow pattern

```python
@graph.node
def draft_change(state: State) -> State:
    state.draft = generate_draft(state)
    return state

@graph.node
def review_change(state: State) -> State:
    state.decision = interrupt({"question": "approve?", "draft": state.draft})
    return state

@graph.node
def apply_if_approved(state: State) -> State:
    if state.decision == "approved":
        apply(state.draft)
    return state

graph.set_entry_point(draft_change)
graph.add_edge(draft_change, review_change)
graph.add_edge(review_change, apply_if_approved)
```

## Related guides

- [Checkpoints](checkpoints.md) — the store used to pause and resume.
- [Observability](observability.md) — `run.paused` / `node.interrupted` events.
