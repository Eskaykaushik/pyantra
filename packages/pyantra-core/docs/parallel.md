# Parallel Execution

Fan out from a node to several branches that run **concurrently**, then
continue at a join node (or end). Each branch executes on an **isolated copy**
of the current state; results are merged back with the field
[reducers](state.md).

```python
from pyantra import Graph

graph = Graph(State)

graph.set_entry_point(ingest)
graph.add_parallel_edges(
    ingest,
    summarize,
    classify,
    extract,
    join=combine,   # optional; default is END
)
```

- `source` runs first.
- `*targets` run concurrently, each on its own copy of the state.
- After every branch completes, their updates are merged back into the shared
  state and execution continues at `join` (a node or `END`).

## How results merge

Branches should **return their updates explicitly** — a branch that returns
`None` contributes nothing.

Unannotated fields are **last-writer-wins**, so concurrent writers to the same
field are racy by nature. Use reducers for fields you want to accumulate:

```python
from typing import Annotated, operator
from dataclasses import dataclass, field

@dataclass
class State:
    query: str = ""
    answers: Annotated[list[str], operator.add] = field(default_factory=list)


@graph.node
def search_web(state: State) -> dict[str, list[str]]:
    return {"answers": ["web result"]}


@graph.node
def search_docs(state: State) -> dict[str, list[str]]:
    return {"answers": ["docs result"]}
```

With `operator.add` as the reducer, both results land in `state.answers`.
Without it, only one would survive.

## Deltas, not full state

A branch may also **mutate its isolated copy in place and return it** instead of
building an update dict:

```python
@graph.node
def search_web(state: State) -> State:
    state.answers.append("web result")
    return state
```

Pyantra diffs each branch's return against the state it received at the fan-out
point, so only the branch's *additions* flow through the reducer. Pre-existing
reducer state is never re-applied, even when the reducer is a plain
non-idempotent function like `operator.add`:

```python
state = State(answers=["existing"])   # fan out from here
# branch 1 appends "web result", branch 2 appends "db result"
result.state.answers == ["existing", "web result", "db result"]
```

Naively merging a mutated copy with `current + value` would re-append the
pre-existing content once per branch. Pyantra avoids that by treating the
returned copy as a delta against the pre-fan-out snapshot. This works for the
canonical list/`operator.add`, set/`operator.or_`, and dict/`merge_dicts`
reducers; for a custom non-invertible reducer, the branch's value is used as
the update in full.

## Full example

```python
import time
from dataclasses import dataclass, field
from typing import Annotated, operator

from pyantra import Graph, RunStatus


@dataclass
class State:
    answers: Annotated[list[str], operator.add] = field(default_factory=list)
    summary: str = ""


def main() -> None:
    graph = Graph(State)

    @graph.node
    def start(state: State) -> dict[str, str]:
        return {"summary": "query received"}

    @graph.node
    def search_web(state: State) -> dict[str, list[str]]:
        time.sleep(0.05)
        return {"answers": ["web result"]}

    @graph.node
    def search_db(state: State) -> dict[str, list[str]]:
        time.sleep(0.05)
        return {"answers": ["db result"]}

    @graph.node
    def join(state: State) -> dict[str, str]:
        return {"summary": "; ".join(state.answers)}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, search_web, search_db, join=join)

    result = graph.compile().run(State())
    assert result.status == RunStatus.COMPLETED
    assert result.state.answers == ["web result", "db result"]
    print(result.state.summary)   # "web result; db result"


if __name__ == "__main__":
    main()
```

## Failures and interrupts

A branch that **fails** fails the whole run: the remaining in-flight branches
are cancelled and awaited before the run reports `FAILED`, so no sibling keeps
running (or emitting events) after the run has failed.

A branch that calls [interrupt](human-in-the-loop.md) pauses the run. Siblings
are cancelled, and the results of branches that **already completed** are merged
into the checkpoint. Resuming later re-enters the fan-out and runs **only the
branches that had not finished** — the interrupted branch and any in-flight
siblings that were cancelled — so completed siblings are never re-run and their
side effects are not duplicated:

```python
app.run(state, checkpointer=store, run_id="order-9")   # pauses inside a branch
app.resume("order-9", "approved", checkpointer=store)   # only unfinished branches run
```

Only in-flight work is lost to cancellation (it is re-run on resume); work that
completed is preserved in the checkpoint.

## Notes

- `add_parallel_edges` requires at least one target and rejects duplicate
  targets (`GraphCompileError` otherwise).
- Branches run concurrently, and all branches must complete before the join
  node runs — except when a branch fails or interrupts, which cancels the rest.
- For async (`async def`) branches, no special handling is needed; sync
  branches run in the same event loop.
- Reducers on the state type are what make merging deterministic — see
  [State merging and reducers](state.md).
