# Graphs and Routing

This guide covers building graphs: nodes, unconditional edges, conditional
routing, and iteration limits. For the basics, see
[Getting started](getting-started.md).

## Nodes

Register a node with the decorator, or with `add_node`:

```python
from pyantra import Graph, NodeConfig

graph = Graph(State)

@graph.node
def load(state: State) -> State:
    return state

# Explicit registration; name is optional (defaults to the function name).
graph.add_node(load, name="loader", config=NodeConfig(retries=2))
```

Nodes may be plain functions or `async def` functions. Both run in either
`run()` or `arun()` — see [Async execution](async.md).

## Unconditional edges

```python
graph.set_entry_point(load)
graph.add_edge(load, transform)
graph.add_edge(transform, END)   # explicit end (optional)
```

If a node has no outgoing edge, execution ends after it.

## Conditional routing

Use `add_conditional_edges` when the next node depends on the state. There are
two forms.

### Form 1 — router returns a key into a `path_map`

```python
def route(state: State) -> str:
    return "positive" if state.value >= 0 else "negative"

graph.set_entry_point(classify)
graph.add_conditional_edges(
    classify,
    route,
    {"positive": process_positive, "negative": process_negative},
)
```

If the router returns a key not present in the map, a `default` target can be
used; without one, execution fails with `InvalidRouteError`:

```python
graph.add_conditional_edges(
    classify,
    route,
    {"positive": process_positive, "negative": process_negative},
    default=fallback,
)
```

### Form 2 — router returns a node name directly

```python
def route(state: State) -> str:
    return "process_positive" if state.value >= 0 else "process_negative"

graph.add_conditional_edges(classify, route)
```

The router may be `async def`; it is awaited when it returns an awaitable.

## Loops and iteration limits

Edges can point backwards, forming a loop:

```python
graph.set_entry_point(extract)
graph.add_conditional_edges(
    extract,
    quality_check,
    {"good": publish, "needs_work": extract},  # re-run extraction
)
```

Loops are bounded by `max_iterations` (default 100). When exceeded, the run
fails with `MaxIterationsError`:

```python
result = app.run(State(), max_iterations=10)
```

## Compile-time validation

`graph.compile()` validates the graph before any execution:

- an entry point is set
- every edge target is a registered node (or `END`)
- router targets exist
- node names are unique

Failures raise `GraphCompileError`. See [Errors](errors.md) for the full
hierarchy.
