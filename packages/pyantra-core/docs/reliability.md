# Reliability

Reliability is configured per node with `NodeConfig`. No configuration means
**fail-fast**: a raised exception fails the node immediately.

```python
from pyantra import Backoff, Graph, NodeConfig

@graph.node
def fetch(state: State) -> State:
    ...

fetch.config = NodeConfig(
    retries=4,                       # retries after the first attempt
    backoff=Backoff.EXPONENTIAL,     # or Backoff.FIXED / Backoff.NONE
    base_delay=1.0,
    max_delay=30.0,
    timeout=15.0,                    # seconds per attempt
)
```

`NodeConfig` can also be passed when registering a node:

```python
graph.add_node(fetch, name="fetch", config=NodeConfig(retries=3))
```

## Field reference

| Field | Default | Meaning |
| --- | --- | --- |
| `retries` | `0` | Number of retries **after** the first attempt. |
| `backoff` | `Backoff.FIXED` | Delay strategy between attempts. |
| `base_delay` | `1.0` | Base delay in seconds. |
| `max_delay` | `None` | Cap on the per-attempt delay. |
| `timeout` | `None` | Per-attempt timeout in seconds. |
| `breaker` | `None` | A `CircuitBreaker` guarding this node. |
| `retry_on` | `None` | Only retry failures matching these exception types. |

## Backoff strategies

`compute_delay(backoff, attempt, base_delay, max_delay)` selects the delay
before a given (1-based) retry attempt:

- `Backoff.NONE` — no delay.
- `Backoff.FIXED` — `base_delay` every attempt.
- `Backoff.EXPONENTIAL` — `base_delay * 2 ** (attempt - 1)`, so 1s, 2s, 4s, 8s…

Both are capped by `max_delay`. A `base_delay <= 0` disables delay.

## Timeouts

`timeout` bounds a single attempt in seconds. On expiry the attempt is
cancelled and the run eventually fails with `NodeTimeoutError` — after retries
are exhausted:

```python
async def slow(state: State) -> State:
    await asyncio.sleep(10)
    return state

slow_node = graph.node(slow, config=NodeConfig(timeout=0.05))
```

See [Async execution](async.md) for how timeouts interact with blocking code.

## Retrying only certain errors

By default, any retryable failure is retried. Use `retry_on` to restrict
retries to specific exception types — anything else fails immediately:

```python
fetch.config = NodeConfig(
    retries=4,
    retry_on=(ConnectionError, TimeoutError),
)
```

A single type is accepted as shorthand: `retry_on=ConnectionError`. Matching is
`isinstance`-based, so subclasses match. `TimeoutError` counts as a retryable
failure, so `retry_on=(TimeoutError,)` retries on timeouts.

## Never retrying certain errors

Errors that must never be retried (bad input, schema violations, …) fail
immediately regardless of the retry policy. Mark them with `@non_retryable`:

```python
from pyantra import non_retryable

@non_retryable
class ValidationError(Exception):
    ...

def fetch(state: State) -> State:
    raise ValidationError("bad request")
```

`is_retryable(exc)` reports whether an exception would be retried. A class or
instance can also opt out by setting `__retryable__ = False`. The
`@non_retryable` marker always wins over `retry_on`.

The base class `NonRetryableError` derives from `PyantraError`; subclass it to
build your own never-retried error types.

## Failure behaviour

- When a node fails and retries remain: a `node.retrying` event is emitted and
  the node runs again.
- When a node fails and retries are exhausted: the run fails with
  `RetryExhaustedError` (when `retries > 0`) or `NodeExecutionError`.
- A timeout after all retries raises `NodeTimeoutError`.

All of these are subclasses of `GraphExecutionError` — see
[Errors](errors.md).

## Circuit breakers

A circuit breaker stops hammering a node after a run of consecutive failures,
then allows a trial call after a reset period. It lives on the node and is
shared across runs:

```python
from pyantra import CircuitBreaker, NodeConfig

breaker = CircuitBreaker(failure_threshold=5, reset_timeout=30.0)

graph.add_node(external_api, name="external_api", config=NodeConfig(breaker=breaker))
```

See [Circuit breakers](circuit-breakers.md) for the full guide.
