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
- **State merging** — per-field reducers (`Annotated[list[T], reducer]`) and partial updates, so concurrent and sequential nodes can contribute to shared state safely.
- **Compile-time validation** — malformed graphs fail early with clear errors, not at runtime.
- **Parallel fan-out** — nodes run concurrently on isolated state copies; in-place branch mutations merge back as deltas, so pre-existing reducer state is never double-counted.
- **Reliability first-class** — per-node retry with backoff, timeouts, and circuit breakers.
- **Checkpoints** — durable snapshots that let failed runs resume where they left off, backed by memory or SQLite.
- **Human-in-the-loop** — `interrupt()` pauses a run for input; `resume()` continues it.
- **Structured observability** — every run produces a rich event trace.
- **LLM abstraction** — a dependency-free provider interface with built-in token/cost tracking.
- **Sync + async** — one traversal engine exposed through both `run()` and `arun()`.
- **Zero dependencies** — pure Python standard library. No databases, no services.

---

## Installation

```bash
pip install pyantra
```

Requires Python 3.10 or later.

---

## Packages

Pyantra is a monorepo of small, focused packages. The core is dependency-free;
integrations opt in only when you install them.

| Package | What it provides |
| --- | --- |
| [`pyantra`](packages/pyantra-core/README.md) | Core workflow engine — graphs, reliability, checkpoints, LLM abstraction. |
| `pyantra-memory` | Vector stores, RAG, and caching. *(planned)* |
| [`pyantra-eval`](packages/pyantra-eval/README.md) | Trajectory evals, LLM judges, pytest plugin. |
| `pyantra-guard` | Runtime type guards, LLM budget caps, PII redaction. |
| `pyantra-otel` | OpenTelemetry & tracing exporters. *(planned)* |
| `pyantra-studio` | Local dev server & visual web debugger. *(planned)* |

```bash
pip install pyantra                    # core only
pip install pyantra pyantra-memory     # core + a companion package
```

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

### State merging and reducers

Annotate a state field with a reducer to control how updates combine instead
of overwriting:

```python
from typing import Annotated, operator
from dataclasses import dataclass, field

@dataclass
class State:
    messages: Annotated[list[str], operator.add] = field(default_factory=list)

@graph.node
def record(state: State) -> dict[str, list[str]]:
    return {"messages": ["hello"]}
```

Nodes may return:

* `None` — the node mutated state in place (reducers do not apply).
* the state type — merged field by field; annotated fields are reduced
  against the current values, all others replace.
* a `dict` of field updates — the same merge, applied per key.

Any `(current, update) -> new` callable works as a reducer; `operator.add`
on lists, `operator.or_` on sets, and dict merges are common. State merge
works for sequential runs and is what makes parallel fan-out safe.

### Parallel execution

Fan out from a node to several branches that run concurrently, then continue
at a join node (or end):

```python
graph.set_entry_point(ingest)
graph.add_parallel_edges(ingest, summarize, classify, join=combine)
```

Each branch executes on an isolated copy of the current state. Results merge
back with the field reducers — unannotated fields are last-writer-wins, so use
reducers for shared fields you want to accumulate.

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

### Retry only certain errors

By default any retryable failure is retried. Use `retry_on` to restrict retries
to specific exception types — anything else fails immediately. This pairs well
with the `@non_retryable` marker, which always wins:

```python
fetch.config = NodeConfig(
    retries=4,
    retry_on=(ConnectionError, TimeoutError),  # only retry these
)
```

A single type is accepted as shorthand: `retry_on=ConnectionError`. Timeouts
count as retryable failures, so `retry_on=(TimeoutError,)` retries on timeouts.

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

## LLMs

Pyantra ships a dependency-free provider abstraction (`LLM`) plus `Message`,
`Usage`, and `LLMResponse` value types. Any model adapter implements
`generate()` / `agenerate()`; providers (OpenAI, Anthropic, …) can live as
extras. Cost and token usage is aggregated per run with `UsageTracker`, and
`MockLLM` provides scripted responses for tests.

```python
from pyantra import Message, MockLLM, UsageTracker

llm = MockLLM(responses=["summarized"], input_tokens=3, output_tokens=2)
tracker = UsageTracker()

def summarize(state: State) -> State:
    resp = llm.generate([Message(role="user", content=state.prompt)])
    tracker.add(resp.usage)
    state.summary = resp.content
    return state
```

`tracker.total` reports aggregate input/output/cache tokens and cost. See
[`docs/llm.md`](packages/pyantra-core/docs/llm.md) for the design and roadmap.

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
and a durable `SQLiteCheckpointStore` is built in:

```python
from pyantra import SQLiteCheckpointStore

store = SQLiteCheckpointStore("checkpoints.db")
```

State, events, and pending interrupts are serialized by the store's pluggable
serializer — JSON by default (safe and portable; state fields and interrupt
payloads must be JSON-serializable). A `PickleSerializer` is available for
arbitrary object graphs; do not use it with checkpoints from untrusted
sources. `SQLiteCheckpointStore` survives process restarts. Additional
backends (Postgres, Redis) can be added behind the same interface.

---

## Human-in-the-loop

Call `interrupt()` from a node to pause a run and request input. The run
pauses with `RunStatus.PAUSED`, its payload lands on `run.interrupt`, and the
state is checkpointed. Resume with `app.resume(...)` — the call to
`interrupt()` then returns the value you provided:

```python
from pyantra import interrupt

@graph.node
def review(state: State) -> State:
    decision = interrupt({"question": "approve this change?", "draft": state.draft})
    state.decision = decision
    return state

run = app.run(state, checkpointer=store, run_id="review-7")
assert run.status == RunStatus.PAUSED
print(run.interrupt)          # {"question": "...", "draft": ...}

resumed = app.resume("review-7", "approved", checkpointer=store)
assert resumed.status == RunStatus.COMPLETED
```

`interrupt()` raises a `BaseException`-derived signal, so a node's own
`except Exception` cannot swallow it. Multiple sequential interruptions in one
run are supported; each `resume()` answers the most recent one.

---

## Integrations

pyantra speaks the surrounding agent ecosystem: MCP servers, A2A peers, and
durable execution frameworks like DBOS. A2A needs nothing extra; MCP and DBOS
are optional extras.

### Model Context Protocol (MCP)

Expose tools from any MCP server as graph nodes — stdio and streamable-HTTP
servers, real or mocked:

```bash
pip install 'pyantra[mcp]'
```

```python
import asyncio
from dataclasses import dataclass, field

from pyantra import END, Graph, McpClient, McpToolNode

@dataclass
class State:
    url: str = ""
    results: list[str] = field(default_factory=list)

async def main():
    client = await McpClient(command="uvx", args=["mcp-server-fetch"]).connect()
    try:
        graph = Graph(State)
        tool = McpToolNode(
            name="fetch",
            client=client,
            tool_name="fetch",
            result_field="results",
            args_from={"url": "url"},
        )
        graph.add_node(tool)
        graph.set_entry_point(tool)
        graph.add_edge(tool, END)
        run = await graph.compile().arun(State(url="https://example.com"))
        print(run.state.results)
    finally:
        await client.close()

asyncio.run(main())
```

`McpToolNode` pulls the tool's JSON schema from the server lazily and validates
each call's arguments against it.

### Agent-to-Agent (A2A)

Delegate work to remote agents over the A2A protocol. pyantra ships a
stdlib-only JSON-RPC client plus `DelegateNode`, which hands off a task, waits
out its lifecycle, and merges the agent's reply into state. No extra install:

```python
from pyantra import A2aClient, DelegateNode, END, Graph
from pyantra.a2a import Message, TextPart

@dataclass
class State:
    prompt: str = ""
    answer: str = ""

client = A2aClient(agent_url="https://agent.example.com/rpc")
graph = Graph(State)
delegate = DelegateNode(
    name="translate",
    client=client,
    result_field="answer",
    payload_from=lambda s: Message(role="user", parts=[TextPart(text=s.prompt)]),
)
graph.add_node(delegate)
graph.set_entry_point(delegate)
graph.add_edge(delegate, END)

run = graph.compile().run(State(prompt="hello"))
print(run.state.answer)
```

When the remote agent asks for input (`input-required`), the run pauses through
pyantra's normal `interrupt()` machinery — set `task_id_field` to keep the
remote task id in state, and `app.resume(run_id, value, checkpointer=...)`
continues the same task. `A2aClient` is a duck-typed protocol, so alternate
transports (and test doubles) can stand in.

### DBOS durable checkpoints

Store checkpoints through a DBOS Transact datasource, so writes ride DBOS's
durability and exactly-once layer:

```bash
pip install 'pyantra[dbos]'
```

```python
from dbos import SQLAlchemyDatasource
from pyantra import DBOSCheckpointStore, Graph

datasource = SQLAlchemyDatasource.create("sqlite:///app.db")
store = DBOSCheckpointStore(datasource=datasource)  # or url="sqlite:///app.db"

app = graph.compile()
run = app.run(state, checkpointer=store, run_id="orders-42")
assert run.status == RunStatus.PAUSED

resumed = app.resume("orders-42", "approved", checkpointer=store)
assert resumed.status == RunStatus.COMPLETED
```

`DBOSCheckpointStore` stores checkpoints in a `pyantra_checkpoints` table and
runs every operation as a datasource transaction — when your graph runs inside
a `@DBOS.workflow`, its checkpoints get DBOS's exactly-once guarantees.

---

## Observability

Every run returns a `Run` object with a structured event trace — no logging
parsing required:

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

Example events:

```
run.started        node.started        node.attempt.failed
run.completed      node.completed      node.attempt.timeout
run.failed         node.failed         node.retrying
run.paused         node.interrupted    edge.selected
run.resumed
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

End-to-end runnable examples live in
[`packages/pyantra-core/examples/`](packages/pyantra-core/examples/):

```bash
python packages/pyantra-core/examples/basic_workflow.py
python packages/pyantra-core/examples/reliability_workflow.py
python packages/pyantra-core/examples/advanced_workflow.py   # reducers, parallel, human-in-the-loop
```

---

## Docs and tutorials

Guides for each core topic live in
[`packages/pyantra-core/docs/`](packages/pyantra-core/docs/):

- [Getting started](packages/pyantra-core/docs/getting-started.md) — install, quickstart, core concepts
- [Graphs and routing](packages/pyantra-core/docs/graphs.md) — nodes, edges, conditional routing, `END`
- [State merging](packages/pyantra-core/docs/state.md) — reducers and partial updates with `Annotated`
- [Parallel execution](packages/pyantra-core/docs/parallel.md) — fan-out/fan-in with reducers
- [Async execution](packages/pyantra-core/docs/async.md) — `arun()`, mixed sync/async nodes
- [Reliability](packages/pyantra-core/docs/reliability.md) — retries, backoff, timeouts, `retry_on`
- [Circuit breakers](packages/pyantra-core/docs/circuit-breakers.md) — thresholds, reset, half-open
- [LLMs](packages/pyantra-core/docs/llm.md) — provider interface, `MockLLM`, `UsageTracker`
- [Checkpoints](packages/pyantra-core/docs/checkpoints.md) — memory and SQLite stores, resume
- [Human-in-the-loop](packages/pyantra-core/docs/human-in-the-loop.md) — `interrupt()` and `resume()`
- [Observability](packages/pyantra-core/docs/observability.md) — event traces and `Run` results
- [Errors](packages/pyantra-core/docs/errors.md) — the `PyantraError` hierarchy

---

## Development

Pyantra uses [`uv`](https://docs.astral.sh/uv/) workspaces. Install uv, then:

```bash
uv sync --all-packages   # install every workspace package

uv run ruff check .                                  # lint
uv run mypy packages/pyantra-core/pyantra            # type check
uv run pytest                                        # test suite
```

This repository uses [conventional commits](https://www.conventionalcommits.org/).

### Releasing

All packages share one version and one tag. To release:

1. Bump `version` in `pyproject.toml` and every `packages/*/pyproject.toml`.
2. Add each package you want shipped to `RELEASE_PACKAGES` in
   `.github/workflows/publish.yml` (placeholders stay out until they have
   real content).
3. Commit, then tag with `v<version>` (e.g. `v0.4.0`) and push. The tag
   triggers the `Publish to PyPI` workflow, which builds every package in
   `RELEASE_PACKAGES` and uploads them to PyPI.

---

## Roadmap

- Automatic LLM usage capture with per-run budgets and compression
- LLM caching and model tiering
- Multi-agent delegation and scoped handoffs
- `interrupt()` defaults and tool/approval-specific helpers
- Deterministic replay of traces

---

## License

[MIT](LICENSE)
