# Issues

## 1. Parallel fan-out double-counts pre-existing reducer state

**Status:** fixed — `_run_parallel` now diffs each state-typed branch result
against the pristine pre-fan-out snapshot via `diff_state`
(`packages/pyantra-core/pyantra/state/reducers.py`) before merging, so reducer
deltas are applied once against the surviving base.

Parallel branches that mutate their isolated state copy and return it (the
pattern documented by `test_parallel_branch_mutates_its_own_copy` in
`packages/pyantra-core/tests/test_parallel.py`) produce corrupted state when
the reducer field already contains data before the fan-out.

### Repro

```python
import operator
from dataclasses import dataclass, field
from typing import Annotated
from pyantra import Graph

@dataclass
class S:
    results: Annotated[list[str], operator.add] = field(default_factory=list)

g = Graph(S)

@g.node
def start(s: S) -> S:
    return s

@g.node
def branch_a(s: S) -> S:
    s.results.append('a')
    return s

@g.node
def branch_b(s: S) -> S:
    s.results.append('b')
    return s

g.set_entry_point(start)
g.add_parallel_edges(start, branch_a, branch_b)
run = g.compile().run(S(results=['base']))

print(run.state.results)
# actual:   ['base', 'base', 'a', 'base', 'b']
# expected: ['base', 'a', 'b']
```

### Root cause

- `_run_branch` (`packages/pyantra-core/pyantra/runtime/executor.py`) deep-copies
  the **full** state, so each branch's returned list already contains the base
  content.
- `_run_parallel` then merges each branch result via `merge_state` /
  `apply_updates`, which runs the reducer as `reducer(current, value)` =
  `current + value`. Since `value` is the full accumulated list, the base is
  concatenated once per branch.

The existing test only passes because the base list is empty; any pre-existing
reducer state (which is the normal case in real workflows) is duplicated. The
same class of bug affects dict-reducer fields and parallel fan-out after a
checkpointed resume. Returning a delta dict avoids it, but the documented
in-place-copy pattern is broken.

## 2. Parallel fan-out orphans sibling branches on failure or interrupt

When one parallel branch fails or interrupts, `asyncio.gather` in
`_run_parallel` (`packages/pyantra-core/pyantra/runtime/executor.py`) propagates
the exception immediately but does **not** cancel the sibling tasks. They keep
executing after the run has already failed/paused.

### Repro (async path)

```python
import asyncio
from dataclasses import dataclass, field
from pyantra import Graph

@dataclass
class S:
    history: list[str] = field(default_factory=list)

g = Graph(S)

@g.node
def start(s: S) -> S:
    return s

@g.node
async def slow_ok(s: S) -> S:
    await asyncio.sleep(0.3)
    return {'history': ['ok']}

@g.node
def boom(s: S) -> S:
    raise RuntimeError('boom')

g.set_entry_point(start)
g.add_parallel_edges(start, slow_ok, boom)

run = await g.compile().arun(S())   # FAILED, returns in ~0.01s
await asyncio.sleep(0.5)            # slow_ok still runs in the background
# run.events: [..., 'run.failed', ..., 'node.completed', 'slow_ok']
#   -- 'node.completed' is appended AFTER 'run.failed'
```

### Consequences

- The failed/paused `Run` returns while sibling branches keep running, so
  their side effects (LLM calls, external API calls) continue and can run to
  completion unreported.
- The event trace ordering is corrupted: a sibling's `node.completed` lands
  after `run.failed` / `run.paused`.
- If a still-running sibling later calls `interrupt()`, the `GraphInterrupt`
  raised in a never-awaited background task produces "Task exception was never
  retrieved" warnings.
- In the sync `.run()` path this is masked because `asyncio.run` cancels all
  pending tasks at loop teardown; it only manifests in `arun()` or any
  embedded event loop.

Related: resuming an interrupt that happened inside a parallel branch
re-executes the whole fan-out (source + all branches), because the checkpoint's
`resume_at` still points at the fan-out source node rather than the interrupted
branch.

## 3. Node-raised `TimeoutError` is misreported as a guard timeout

`_invoke_with_policy` (`packages/pyantra-core/pyantra/runtime/executor.py`)
catches `(asyncio.TimeoutError, TimeoutError)` regardless of whether the timeout
came from the `with_timeout` guard or was raised by the node itself, and always
converts it to `NodeTimeoutError`.

### Repro

```python
from dataclasses import dataclass
from pyantra import Graph, NodeConfig

@dataclass
class S:
    value: int = 0

g = Graph(S)

@g.node(config=NodeConfig(retries=2))  # no timeout configured
def fetch(s: S) -> S:
    raise TimeoutError('the HTTP request timed out')  # node's own error

g.set_entry_point(fetch)
run = g.compile().run(S())

print(run.status.value)      # failed
print(run.error)             # "Node 'fetch' exceeded timeout of Nones."
print(type(run.exception).__name__)  # NodeTimeoutError
```

### Consequences

- The error message reads `exceeded timeout of Nones.` when `config.timeout`
  is `None`.
- A node's own `TimeoutError` is treated as retryable by default (`retry_on`
  unset), so it is retried and finally surfaced as a misleading
  `NodeTimeoutError` instead of a `NodeExecutionError` describing the node's
  actual failure.

## 4. `__version__` drift between packages and the workspace

`packages/pyantra-core/pyantra/__init__.py` reports `__version__ = "0.5.0"`,
while the root `pyproject.toml` and the released packages are on `0.5.1`.
`pyantra.__version__` will be stale for the current release.

## 5. `pyantra-guard` `typecheck` crashes on bare (unparameterized) generics

`typecheck` (`packages/pyantra-guard/pyantra_guard/typeguard.py`) unpacks
`typing.get_args(expected)` unconditionally for `list`/`set`/`dict`/`tuple`,
so bare generics (no type arguments) raise instead of returning a verdict.

### Repro

```python
from typing import Dict, List
from pyantra_guard.typeguard import typecheck

typecheck({}, typing.Dict)   # ValueError: not enough values to unpack (expected 2, got 0)
typecheck([1], typing.List)  # IndexError: tuple index out of range
```

### Consequences

A dataclass state field annotated with a bare generic (`typing.List`,
`typing.Dict`, `typing.Set`, `typing.Tuple` — not the `list[...]` forms) makes
`check_state`/`assert_state` raise a crash instead of validating the value.

## 6. `BudgetTracker.record` commits usage before the budget check

`BudgetTracker.record` (`packages/pyantra-guard/pyantra_guard/budget.py`) adds
the call's usage to the running total first, then checks the budget. When
`BudgetError` is raised the violating usage is already counted.

### Repro

```python
tracker = BudgetTracker(Budget(max_total_tokens=5))
tracker.record(Usage(input_tokens=10))   # raises BudgetError
tracker.total.total_tokens               # -> 10  (violating record still counted)
```

### Consequences

If a caller catches `BudgetError` and continues, the cap is effectively
bypassed (the excess stays in the total), and the tracker's reported aggregate
usage overstates what actually ran.

## 7. Resuming a parallel-branch interrupt replays the whole fan-out

Checkpoints are written only at node boundaries with `resume_at` set to the
fan-out **source** node; no progress is recorded inside branches. So
`app.resume()` after a pause that happened inside a parallel branch re-runs the
source **and every branch**, including branches that already completed with
external side effects.

### Repro

```python
from dataclasses import dataclass, field
from pyantra import Graph, MemoryCheckpointStore, interrupt

@dataclass
class S:
    out: list[str] = field(default_factory=list)
    decision: str = ""

calls = {"side_effect": 0}
store = MemoryCheckpointStore()
g = Graph(S)

@g.node
def start(s: S) -> S:
    return s

@g.node
def side_effect(s: S) -> dict:
    calls["side_effect"] += 1
    return {"out": ["result"]}

@g.node
def interrupted(s: S) -> dict:
    decision = interrupt("go?")
    return {"decision": decision}

g.set_entry_point(start)
g.add_parallel_edges(start, side_effect, interrupted)
app = g.compile()

app.run(S(), checkpointer=store, run_id="par2")     # paused; side_effect ran once
app.resume("par2", "yes", checkpointer=store)       # side_effect runs AGAIN
```

### Consequences

External side effects (LLM calls, API calls) made by already-completed sibling
branches are duplicated on every resume. This is distinct from issue 2: it
happens with well-behaved branches, purely from the checkpoint resume
granularity for parallel execution.

## 8. Interrupt payloads containing `set`/`tuple` fail durable checkpoint saves

`JsonSerializer.dumps` (`packages/pyantra-core/pyantra/checkpoint/serializer.py`)
applies `_jsonify` (which converts sets/tuples to lists) to state, but passes
`interrupts` straight to `json.dumps` untouched.

### Repro

```python
from pyantra import Graph, SQLiteCheckpointStore, interrupt

@g.node
def review(s: S) -> dict:
    decision = interrupt({"choices": {"a", "b"}})  # set in the payload
    return {"decision": decision}

app = g.compile()
store = SQLiteCheckpointStore("checkpoints.db")
run = app.run(S(), checkpointer=store, run_id="int-payload")
# run.status == FAILED (json.dumps TypeError on the set)
```

### Consequences

A run that pauses with a set/tuple in its interrupt payload fails when the
checkpoint is persisted through a serializing store (SQLite, DBOS), even
though `_jsonify` exists precisely to make such values portable and
`MemoryCheckpointStore` (which does not serialize) handles the same run fine.

## 9. Persisted checkpoint traces lose the events that caused the pause or failure

`Executor._checkpoint` (`packages/pyantra-core/pyantra/runtime/executor.py`) saves
`events=list(run.events)` **before** the next node runs, and the checkpoint is
never rewritten afterward. `_persist_interrupt` appends the interrupt entry but
not the events emitted by the failing/pausing attempt. So `node.interrupted`,
`node.failed`, and `run.paused` never reach the durable checkpoint.

### Repro

```python
from dataclasses import dataclass
from pyantra import Graph, MemoryCheckpointStore, interrupt

@dataclass
class S:
    x: int = 0

g = Graph(S)

@g.node
def a(s: S) -> S:
    interrupt("ask")
    return s

g.set_entry_point(a)
store = MemoryCheckpointStore()
run = g.compile().run(S(), checkpointer=store, run_id="t1")

print([e.event for e in run.events])
# ['run.started', 'node.started', 'node.interrupted', 'run.paused']
print([e.event for e in store.load("t1").events])
# ['run.started']   -- the reason for the pause is gone
```

### Consequences

A resumed run's event trace (and `run.usage` reconstruction from it) silently
loses why the run paused or failed, so observability and cost attribution are
incomplete for any checkpointer-backed resume. The same applies to `node.failed`
for failed runs.

## 10. Resuming a parallel-branch interrupt replays the whole fan-out and double-bills siblings

`aresume` (`packages/pyantra-core/pyantra/graph/compiler.py`) computes the resume
node from `checkpoint.interrupts[-1][0]`, but `Executor.arun`
(`packages/pyantra-core/pyantra/runtime/executor.py`) then re-derives the actual
start position from `checkpoint.resume_at` — the fan-out **source** node. The
node chosen by `aresume` is used only as the interrupt-response key. So resuming
an interrupt raised inside a parallel branch re-runs the source **and every
branch**, including siblings that already completed.

### Repro

```python
from dataclasses import dataclass, field
from pyantra import Graph, MemoryCheckpointStore, interrupt

@dataclass
class S:
    out: list[str] = field(default_factory=list)
    decision: str = ""

calls = {"side_effect": 0}
store = MemoryCheckpointStore()
g = Graph(S)

@g.node
def start(s: S) -> S:
    return s

@g.node
def side_effect(s: S) -> dict:
    calls["side_effect"] += 1
    return {"out": ["result"]}

@g.node
def interrupted(s: S) -> dict:
    decision = interrupt("go?")
    return {"decision": decision}

g.set_entry_point(start)
g.add_parallel_edges(start, side_effect, interrupted)
app = g.compile()

app.run(S(), checkpointer=store, run_id="p1")   # paused; side_effect ran once
app.resume("p1", "yes", checkpointer=store)     # side_effect runs AGAIN
```

### Consequences

Distinct mechanism from issue 7, which it overlaps with: `aresume`'s selected
node never becomes the resume position, so the two can disagree, and completed
sibling branches re-execute their external side effects (LLM calls, API calls).
That replay is billed twice. If the fan-out source's outgoing edges were ever
conditional, the interrupted branch could be skipped entirely and the resume
value silently dropped, leaving the run paused.

## 11. Duplicate parallel targets are not validated

`Graph.add_parallel_edges` (`packages/pyantra-core/pyantra/graph/graph.py`) and
`validate` (`packages/pyantra-core/pyantra/graph/compiler.py`) accept the same
node as multiple fan-out targets. Each target runs its own task on its own
deep copy, so the branch's result is merged once per occurrence.

### Repro

```python
from dataclasses import dataclass
from typing import Annotated
from pyantra import Graph

@dataclass
class S:
    v: Annotated[list[int], lambda c, u: c + u]

g = Graph(S)

@g.node
def start(s: S) -> S:
    return s

@g.node
def leaf(s: S) -> dict:
    return {"v": [1]}

g.set_entry_point(start)
g.add_parallel_edges(start, leaf, leaf)   # compiles without error
run = g.compile().run(S(v=[]))
print(run.state.v)                        # [1, 1] -- same branch applied twice
```

### Consequences

A reducer field receives the duplicate branch's update once per occurrence
(`[1, 1]` instead of `[1]`), silently corrupting state. The duplicate target
should be rejected at compile time (or documented as an error).

## 12. Non-dataclass state cannot be returned as a fresh instance

`merge_state` (`packages/pyantra-core/pyantra/state/reducers.py`) calls
`dataclasses.fields(returned)` unconditionally when a node returns the state
type. For a non-dataclass state type, returning a **new** instance raises
`TypeError`, which `_merge_result` wraps into `NodeExecutionError` and the run
fails. Only in-place mutation works, despite the documented contract that "any
object may act as workflow state".

### Repro

```python
from pyantra import Graph

class Plain:
    def __init__(self, x: int = 0) -> None:
        self.x = x

g = Graph(Plain)

@g.node
def inc(s: Plain) -> Plain:
    return Plain(s.x + 1)   # fresh instance, not in-place

g.set_entry_point(inc)
run = g.compile().run(Plain())
print(run.status.value)     # failed
print(run.error)            # "Node 'inc' returned state that failed to merge: ..."
```

### Consequences

The documented "any object may act as workflow state" path breaks as soon as a
node constructs a new state object instead of mutating in place — the common
pattern for immutable or value-style states.

## 13. `Usage.__add__` keeps the first model label

`Usage.__add__` (`packages/pyantra-core/pyantra/llm/types.py`) sets
`model=self.model or other.model`, so when aggregating usage across calls that
used different models, the first non-empty model name wins and later models are
silently dropped from the aggregate.

### Repro

```python
from pyantra import Usage

a = Usage(input_tokens=10, model="mock-a")
b = Usage(input_tokens=10, model="mock-b")
print((a + b).model)   # "mock-a" -- the mixed aggregate is mislabeled
```

### Consequences

Per-run cost attribution (and any report keyed off `Run.usage.model`) is wrong
for heterogeneous runs; the summed `cost` cannot be matched to its model.

## 14. A2A client leaks raw stdlib network errors

`A2aClient._post` and `_get_json` (`packages/pyantra-core/pyantra/a2a/client.py`)
wrap `urllib.error.HTTPError` into `A2aError` but leave `urllib.error.URLError`,
`TimeoutError`, and `OSError` (DNS failure, connection refused, socket timeout)
uncaught.

### Consequences

`DelegateNode` and callers observe raw stdlib exceptions instead of the
documented `A2aError` contract, so failure handling in nodes and retry policies
cannot reliably key on `A2aError`.

## 15. `pyantra-guard` `typecheck` crashes on `Literal` types

`typecheck` (`packages/pyantra-guard/pyantra_guard/typeguard.py`) falls through
to `isinstance(value, origin)` for any origin it does not special-case. For
`typing.Literal[...]` (and other non-type origins) `isinstance` raises a
`TypeError` instead of returning a verdict — a crash path distinct from issue 5's
bare-generic unpacking.

### Repro

```python
from typing import Literal
from pyantra_guard.typeguard import typecheck

typecheck("a", Literal["a", "b"])
# TypeError: typing.Literal cannot be used with isinstance()
```

### Consequences

A state field annotated with a `Literal` type makes `check_state`/`assert_state`
raise a crash instead of validating the value.

## 16. `PIIRedactor.redact_value` misses PII in dict keys

`PIIRedactor.redact_value` (`packages/pyantra-guard/pyantra_guard/redaction.py`)
redacts string **values** recursively but leaves dict **keys** untouched.

### Repro

```python
from pyantra_guard import redact_value

redact_value({"user@example.com": "call 555-0100"})
# {'user@example.com': 'call <phone>'}   -- the email key is still raw
```

### Consequences

PII that appears as a dict key (e.g. an email or IP used as a lookup key in
state, interrupt payloads, or tool output) survives redaction and leaks into
redacted traces.
