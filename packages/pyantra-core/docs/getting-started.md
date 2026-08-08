# Getting Started

Pyantra is a Python framework for building AI agent workflows as **typed graphs
of nodes and edges**. It is small, composable, and dependency-free, with
reliability and observability built in by default.

This guide covers installation and the core concepts. For the full topic list
see the [README](../../../README.md#docs-and-tutorials).

## Requirements

- Python 3.10 or later
- No third-party dependencies — only the standard library

## Installation

```bash
pip install pyantra
```

To install for development (lint, type checks, tests):

```bash
pip install -e ".[dev]"
```

## Quickstart

A workflow is a **graph** of **nodes** connected by **edges**. Nodes receive
state and return updated state. The graph is compiled — and validated — before
it can be executed.

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

## Core concepts

### State

Any object can act as state; a `@dataclass` is recommended. Pyantra uses the
type to validate the flow of state between nodes and to merge field updates
(see [State merging](state.md)).

### Nodes

Nodes are plain functions — sync or `async` — that receive the state and return
an update:

- `None` — the node mutated state in place.
- the state type — merged field by field.
- a `dict[str, Any]` of field updates — merged per key.

Register a node with the `@graph.node` decorator, or explicitly:

```python
graph.add_node(fn, name="fetch", config=NodeConfig(retries=3))
```

### Edges

Edges connect nodes. A node runs after all its incoming edges resolve:

```python
graph.set_entry_point(ingest)
graph.add_edge(ingest, transform)
graph.add_edge(transform, output)
```

A node with no outgoing edge ends the workflow. You can also be explicit:

```python
from pyantra import END

graph.add_edge(final_node, END)
```

### Compile, then run

`graph.compile()` validates the graph and returns a `CompiledGraph`:

```python
app = graph.compile()
result = app.run(State(value=1))   # synchronous
result = await app.arun(State(value=1))  # asynchronous
```

`run()` returns a `Run` object with the final state, status, and a structured
event trace — see [Observability](observability.md).

## What's next

- [Graphs and routing](graphs.md) — conditional edges, loops, `max_iterations`
- [State merging](state.md) — reducers and partial updates
- [Parallel execution](parallel.md) — concurrent branches
- [Async execution](async.md) — `arun()`
- [Reliability](reliability.md) — retries, timeouts, and more
- [Checkpoints](checkpoints.md) — durable resume
- [Human-in-the-loop](human-in-the-loop.md) — `interrupt()` and `resume()`
