# Circuit Breakers

A **circuit breaker** stops a node from being hammered after a run of
consecutive failures, then allows a trial call once a reset period has elapsed.
This protects slow or failing dependencies (external APIs, databases) and fails
fast when they are known to be down.

## Basic usage

```python
from pyantra import CircuitBreaker, NodeConfig

breaker = CircuitBreaker(failure_threshold=5, reset_timeout=30.0)

graph.add_node(
    external_api,
    name="external_api",
    config=NodeConfig(breaker=breaker),
)
```

## States

A breaker moves through three states:

```
CLOSED ──(failure_threshold consecutive failures)──▶ OPEN
OPEN ──(reset_timeout elapses)──▶ HALF_OPEN
HALF_OPEN ──(success)──▶ CLOSED
HALF_OPEN ──(failure)──▶ OPEN   (immediately, resets the timer)
```

- **CLOSED** — normal operation. Failures accumulate; a success resets the
  count.
- **OPEN** — execution is refused with `CircuitOpenError`; the node never runs.
- **HALF_OPEN** — after the reset period, exactly one trial call is allowed.

`breaker.state` reports the current `CircuitState`:

```python
from pyantra import CircuitState

assert breaker.state in {
    CircuitState.CLOSED,
    CircuitState.OPEN,
    CircuitState.HALF_OPEN,
}
```

## Shared across runs

Breaker state lives on the `CircuitBreaker` object, not on a single run. If you
reuse the same breaker across `run()` calls, failures accumulate and the
circuit opens across runs:

```python
breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
app = graph.compile()

app.run(state)   # 1 failure
app.run(state)   # 2 failures -> circuit opens
third = app.run(state)
assert third.status == RunStatus.FAILED
assert isinstance(third.exception, CircuitOpenError)
```

The node itself was never invoked on the third run, which is the point: the
dependency is protected from further calls while it is down.

## Introspection

```python
breaker.state                 # CircuitState.CLOSED / OPEN / HALF_OPEN
breaker.consecutive_failures  # count of consecutive failures
```

## Manual control

- `breaker.reset()` — close the circuit and clear the failure count
  immediately.
- `breaker.record_success()` — record a success; returns `True` if the circuit
  was open and is now closed.
- `breaker.record_failure()` — record a failure; returns `True` if the circuit
  just opened.

These are used internally by the executor; you only need them for testing or
tooling.

## Notes

- `failure_threshold` must be `>= 1` (`ValueError` otherwise).
- A refused execution surfaces as `CircuitOpenError`, a subclass of
  `GraphExecutionError` — see [Errors](errors.md).
- Combine with retries if you want a few retry attempts before the breaker
  opens: `NodeConfig(retries=2, breaker=breaker)`.
