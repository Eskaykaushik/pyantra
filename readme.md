<p align="center">
  <h1 align="center">Pyantra</h1>
  <p align="center">Typed, observable, reliable workflows for production AI agents.</p>
</p>

<div align="center">

[![CI](https://github.com/Eskaykaushik/pyantra/actions/workflows/ci.yml/badge.svg)](https://github.com/Eskaykaushik/pyantra/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyantra.svg)](https://pypi.org/project/pyantra/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyantra.svg)](https://pypi.org/project/pyantra/)
[![License: MIT](https://img.shields.io/pypi/l/pyantra.svg)](https://opensource.org/licenses/MIT)

</div>

Pyantra is a Python framework for building AI agent workflows as **typed graphs of nodes and edges**. It is designed to be small, composable, and dependency-free, with reliability and observability built in by default.

> **Agents should be observable, reproducible, testable, reliable, and cost-aware by default.**

---

## Features

- **Typed workflows** — nodes flow state through the graph and are type-checked end to end.
- **Compile-time validation** — malformed graphs fail early with clear errors, not at runtime.
- **Reliability first-class** — per-node retry with backoff, timeouts, and circuit breakers.
- **Checkpoints** — durable snapshots that let failed runs resume where they left off.
- **Structured observability** — every run produces a rich event trace.
- **Sync + async** — one traversal engine exposed through both `run()` and `arun()`.
- **Zero dependencies** — pure Python standard library. No databases, no services.

---

## Installation

```bash
pip install pyantra
```

Requires Python 3.10 or later.

---

## Quickstart

```python
from dataclasses import dataclass

from pyantra import Graph


@dataclass
class State:
    value: int


graph = Graph(State)


@graph.node
def increment(state: State) -> State:
    state.value += 1
    return state


@graph.node
def double(state: State) -> State:
    state.value *= 2
    return state


graph.set_entry_point(increment)
graph.add_edge(increment, double)

app = graph.compile()

result = app.run(State(value=1))

assert result.state.value == 4
```

---

## Core concepts

A workflow is a **graph** of **nodes** connected by **edges**. Nodes receive state
and return updated state (or mutate it in place and return `None`). The graph is
compiled — and validated — before it can be executed.

### Conditional routing

```python
@graph.node
def classify(state: State) -> State:
    return state

@graph.node
def process_positive(state: State) -> State:
    ...

@graph.node
def process_negative(state: State) -> State:
    ...

def route(state: State) -> str:
    return "positive" if state.value >= 0 else "negative"

graph.set_entry_point(classify)
graph.add_conditional_edges(
    classify,
    route,
    {"positive": process_positive, "negative": process_negative},
)
```

Nodes can also terminate a workflow explicitly:

```python
from pyantra import END

graph.add_edge(final_node, END)
```

### Async execution

The exact same graph runs asynchronously:

```python
result = await app.arun(State(value=1))
```

Async nodes (`async def`) are supported seamlessly in both modes.

---

## Reliability

Reliability is configured per node with `NodeConfig`. No configuration means
fail-fast: a raised exception fails the node immediately.

```python
from pyantra import Backoff, Graph, NodeConfig

graph = Graph(State)

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

graph.set_entry_point(fetch)
```

`NodeConfig` can also be passed directly when registering a node:

```python
graph.add_node(fetch, name="fetch", config=NodeConfig(retries=3))
```

### Never retry certain errors

Errors that must not be retried (bad input, schema violations, …) can be marked
explicitly — they fail immediately regardless of the retry policy:

```python
from pyantra import non_retryable

@non_retryable
class ValidationError(Exception):
    ...

def fetch(state: State) -> State:
    raise ValidationError("bad request")
```

### Circuit breakers

A circuit breaker stops hammering a node after a run of consecutive failures,
then allows a trial call once a reset period elapses:

```python
from pyantra import CircuitBreaker, NodeConfig

breaker = CircuitBreaker(failure_threshold=5, reset_timeout=30.0)

graph.add_node(
    external_api,
    name="external_api",
    config=NodeConfig(breaker=breaker),
)
```

---

## Checkpoints and resume

Pass a checkpoint store to `run()` and a run can resume from its last
successful node after a failure:

```python
from pyantra import MemoryCheckpointStore

store = MemoryCheckpointStore()

first = app.run(state, checkpointer=store, run_id="order-123")
assert first.status == RunStatus.FAILED

# Re-run with the same run_id: resumes where it stopped instead of restarting.
second = app.run(state, checkpointer=store, run_id="order-123")
```

`CheckpointStore` is an abstract interface; in-memory storage ships by default
and durable backends (SQLite, Postgres, Redis) can be added behind the same API.

---

## Observability

Every run returns a `Run` object with a structured event trace — no logging
parsing required:

```python
result = app.run(state)

result.run_id      # unique id for the run
result.status      # RunStatus (pending, running, completed, failed, ...)
result.state       # final (or last known) state
result.events      # ordered list of RunEvent
result.error       # human-readable failure message, when failed
result.exception   # the underlying exception, when failed
```

Example events:

```
run.started        node.started      node.attempt.failed
run.completed      node.completed    node.attempt.timeout
run.failed         node.failed       node.retrying
run.resumed        edge.selected
```

---

## Errors

All exceptions derive from `PyantraError`:

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

---

## Examples

End-to-end runnable examples live in [`examples/`](examples/):

```bash
python examples/basic_workflow.py
python examples/reliability_workflow.py
```

---

## Development

```bash
pip install -e ".[dev]"

ruff check .          # lint
mypy pyantra          # type check
pytest                # test suite
```

This repository uses [conventional commits](https://www.conventionalcommits.org/).

---

## Roadmap

- Context management with token budgets and compression
- LLM usage tracking, caching, and model tiering
- Multi-agent delegation and scoped handoffs
- Human-in-the-loop pause/resume
- Deterministic replay and trace-based regression testing

---

## License

[MIT](LICENSE)
