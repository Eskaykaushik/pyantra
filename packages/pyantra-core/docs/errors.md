# Errors

All Pyantra exceptions derive from `PyantraError`, so a single `except
PyantraError` catches everything the framework raises.

```
PyantraError
├── GraphCompileError      — the graph failed validation at compile time
├── GraphExecutionError
│   ├── NodeExecutionError — a node raised during execution
│   ├── NodeTimeoutError   — a node exceeded its configured timeout
│   ├── RetryExhaustedError— retries were exhausted
│   ├── CircuitOpenError   — a circuit breaker refused execution
│   ├── InvalidRouteError  — a router returned an unknown destination
│   └── MaxIterationsError — a run exceeded max_iterations
├── CheckpointError        — checkpoint storage or resume failed
└── NonRetryableError      — base class for never-retried errors
```

## Where errors surface

### Compile time

`graph.compile()` raises `GraphCompileError` for malformed graphs — no entry
point, an unknown edge target, a duplicate node name, or a router target that
does not exist. This is deliberate: bad graphs fail early, not at runtime.

```python
from pyantra import Graph, GraphCompileError

try:
    app = graph.compile()
except GraphCompileError as exc:
    print(exc)
```

### Runtime — inside a run

Node failures do **not** raise out of `run()` / `arun()`. They are captured on
the `Run` object:

```python
result = app.run(state)

if result.status == RunStatus.FAILED:
    print(result.error)          # human-readable message
    print(type(result.exception))  # the underlying exception
```

`result.exception` is the raised error — a `GraphExecutionError` subclass such
as `NodeExecutionError`, `RetryExhaustedError`, or `NodeTimeoutError`. The
original exception (the one your node raised) is available as
`result.exception.__cause__`.

```python
if isinstance(result.exception, NodeExecutionError):
    assert isinstance(result.exception.__cause__, ValueError)
```

### Execution errors

| Error | When |
| --- | --- |
| `NodeExecutionError` | A node raised during execution. |
| `NodeTimeoutError` | A node exceeded its configured `timeout`. |
| `RetryExhaustedError` | A node failed after exhausting its retry policy. |
| `CircuitOpenError` | A node's circuit breaker refused execution. |
| `InvalidRouteError` | A router returned an unknown destination. |
| `MaxIterationsError` | A run exceeded `max_iterations`. |

Node-scoped errors carry a `node` attribute (and a `run_id`):

```python
assert result.exception.node == "fetch"   # type: ignore[union-attr]
```

### Checkpoint errors

Checkpoint storage or resume failures raise `CheckpointError` — including when
you call `resume()` for a run with no saved checkpoint.

## Never-retried errors

`NonRetryableError` is the base class for exceptions that must never be
retried. The `@non_retryable` decorator sets the `__retryable__ = False` flag
on a class instead. See [Reliability](reliability.md).

```python
from pyantra import NonRetryableError

class SchemaError(NonRetryableError):
    pass
```

## Catching everything

```python
try:
    result = app.run(state)
    if result.status == RunStatus.FAILED:
        raise result.exception
except PyantraError as exc:
    # compile errors, checkpoint errors, and (re-raised) run failures
    ...
```

## Related guides

- [Observability](observability.md) — inspecting failures on the `Run` object.
- [Reliability](reliability.md) — retry- and timeout-related errors.
