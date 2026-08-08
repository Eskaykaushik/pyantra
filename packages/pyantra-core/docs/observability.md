# Observability

Every run returns a `Run` object with a structured event trace — no log parsing
required. This makes workflows testable and inspectable by default.

## The Run object

```python
result = app.run(state)

result.run_id      # unique id for the run
result.status      # RunStatus (pending, running, completed, failed, paused, ...)
result.state       # final (or last known) state
result.events      # ordered list of RunEvent
result.error       # human-readable failure message, when failed
result.exception   # the underlying exception, when failed
result.interrupt   # the human-in-the-loop payload, when paused
```

`Run` is a `@dataclass`; `run_id` is unique per run unless you pass one.

## RunStatus

`RunStatus` is a `str`-backed enum:

| Value | Meaning |
| --- | --- |
| `PENDING` | Not started yet. |
| `RUNNING` | Currently executing. |
| `COMPLETED` | Finished successfully. |
| `FAILED` | A node (or validation) failed the run. |
| `PAUSED` | Paused for human input via `interrupt()`. |
| `CANCELLED` | Cancelled. |

## Events

`result.events` is an ordered list of `RunEvent`:

```python
from pyantra import RunEvent

# RunEvent fields:
#   run_id       — which run produced this event
#   event        — the event name
#   timestamp    — epoch seconds
#   node         — the node, when node-scoped
#   duration_ms  — node duration, when available
#   message      — optional detail
```

Event names include:

```
run.started        node.started        node.attempt.failed
run.completed      node.completed      node.attempt.timeout
run.failed         node.failed         node.retrying
run.paused         node.interrupted    edge.selected
run.resumed
```

Filter to node-scoped events with `result.node_events`:

```python
node_events = result.node_events   # only events with a node name
```

## Serializing

Both `Run` and `RunEvent` provide `to_dict()` for JSON-friendly output:

```python
payload = {
    "run_id": result.run_id,
    "status": result.status.value,
    "state": result.state,
    "events": [e.to_dict() for e in result.events],
    "error": result.error,
    "interrupt": result.interrupt,
}
```

## Using events in tests

The event trace makes assertions easy:

```python
assert result.status == RunStatus.COMPLETED
assert "node.completed" in {e.event for e in result.events}
assert "node.retrying" in {e.event for e in result.events}
```

## Related guides

- [Errors](errors.md) — failures and `result.exception` / `result.error`.
- [Reliability](reliability.md) — `node.attempt.*` and `node.retrying` events.
- [Human-in-the-loop](human-in-the-loop.md) — `run.paused` / `run.interrupt`.
