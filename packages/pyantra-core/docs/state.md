# State Merging and Reducers

By default a node's returned values replace the corresponding state fields.
**Reducers** let you control how updates combine instead — for example, appending
to a list rather than overwriting it.

This is what makes [parallel fan-out](parallel.md) safe: concurrent branches can
each contribute to shared fields without clobbering one another.

## How updates merge

A node can return three things:

| Return value | Behaviour |
| --- | --- |
| `None` | The node mutated state in place; no merging happens. |
| the state type | Merged field by field: annotated fields are reduced, others are replaced. |
| `dict[str, Any]` | The same merge, applied per key. Unknown keys raise `KeyError`. |

In [parallel fan-out](parallel.md), a branch that mutates its isolated copy and
returns it is merged as a **delta**: the executor diffs the copy against the
pre-fan-out state, so a reducer field's pre-existing content is never
re-applied — even with non-idempotent reducers like `operator.add`.

## Annotating fields with reducers

Use `typing.Annotated` to attach a reducer to a field. Any callable
`(current, update) -> new` works.

```python
from typing import Annotated, operator
from dataclasses import dataclass, field

@dataclass
class State:
    messages: Annotated[list[str], operator.add] = field(default_factory=list)
    seen: Annotated[set[str], operator.or_] = field(default_factory=set)
    value: int = 0
```

Now a node can return partial updates and they accumulate:

```python
@graph.node
def add_message(state: State) -> dict[str, list[str]]:
    return {"messages": ["hello"]}

@graph.node
def add_another(state: State) -> dict[str, list[str]]:
    return {"messages": ["world"]}
```

Running `add_message` then `add_another` leaves `state.messages ==
["hello", "world"]`, because `operator.add` combines the lists instead of
replacing them.

The same works when a node returns a full state object:

```python
@graph.node
def enrich(state: State) -> State:
    state2 = State()
    state2.messages = ["extra"]
    return state2   # messages reduced onto current, value untouched
```

## Dict and set reducers

`operator.or_` merges sets, and a deep merge helper (`merge_dicts`) merges
top-level dict keys instead of replacing the whole dict:

```python
from typing import Annotated, operator
from pyantra.state.reducers import merge_dicts

@dataclass
class State:
    tags: Annotated[set[str], operator.or_] = field(default_factory=set)
    counts: Annotated[dict[str, int], merge_dicts] = field(default_factory=dict)
```

Any callable works, so you can define your own:

```python
def keep_longest(current: list[str], update: list[str]) -> list[str]:
    return update if len(update) >= len(current) else current
```

## Notes

- Reducers are extracted from `Annotated` metadata at compile time (via
  `get_type_hints`), and read once when the graph is built.
- A reducer is only applied when a field update flows through the merge. If a
  node returns `None` after mutating state in place, reducers are not involved.
- Returning a `dict` with an unknown field name raises `KeyError` during
  execution, which becomes a `NodeExecutionError` on the run.

## Custom reducers in detail

Durable [checkpoint stores](checkpoints.md) serialize state with the store's
serializer — JSON by default. With the default serializer, state field values
must be JSON-serializable (primitives, containers, and nested dataclasses);
use `PickleSerializer` for arbitrary objects.
