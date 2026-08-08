# Async Execution

The exact same graph runs asynchronously. Declare nodes as `async def` and call
`arun()` — sync and async nodes mix freely in either mode.

## Running a graph asynchronously

```python
result = await app.arun(State(value=1))
```

That is the only change from the sync `run()` call — same graph, same state
type, same `Run` result.

## Sync and async nodes mixed

```python
@graph.node
def load(state: State) -> State:
    # synchronous work
    return state

@graph.node
async def call_api(state: State) -> State:
    # async I/O
    await asyncio.sleep(0.01)
    return state

@graph.node
def save(state: State) -> State:
    return state
```

All three run correctly in both `run()` and `arun()`. Under the hood `run()`
runs the async engine on a fresh event loop (`asyncio.run`), so you never need
an event loop of your own to use the sync API.

## Async routers

Conditional-edge routers may also be `async def`:

```python
async def route(state: State) -> str:
    await asyncio.sleep(0.01)
    return "positive" if state.value >= 0 else "negative"
```

## Resuming asynchronously

`resume()` has an async twin, `aresume()` — see
[Human-in-the-loop](human-in-the-loop.md):

```python
result = await app.aresume(run_id, "approved", checkpointer=store)
```

## Timeouts in async nodes

Per-node `timeout` in `NodeConfig` applies to both sync and async nodes. For
async nodes the underlying task is cancelled on timeout; cooperative
(`await`-based) code is interrupted cleanly. Synchronous blocking work is not
interrupted until it returns control to the event loop. See
[Reliability](reliability.md).

## Notes

- Don't call `arun()` from inside a node of the same run — the executor already
  owns the event loop.
- Parallel branches run concurrently only under `arun()` (or the event loop
  created by `run()`); see [Parallel execution](parallel.md).
