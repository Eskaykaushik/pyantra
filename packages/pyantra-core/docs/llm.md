# LLM Abstraction — Design Notes

Status: core interface implemented. Automatic usage capture deferred to
Phase 3 (see below).

## Goal

Provider-agnostic model calls that are observable and cost-controlled by
default, without coupling the core to OpenAI, Anthropic, etc.

## What ships now

- `Message` — provider-agnostic role/content type.
- `Usage` — token + cost accounting for one call (`total_tokens`, `__add__`
  for aggregation).
- `LLMResponse` — content + usage.
- `LLM` — minimal protocol (`generate` / `agenerate`); provider adapters
  implement it. The core stays dependency-free.
- `MockLLM` — scripted, deterministic responses with fixed token counts, for
  tests, examples, and future deterministic replay. Records the messages it
  receives on `recorded_calls`.
- `UsageTracker` — explicit per-run accumulation (`add` / `total` / `reset`),
  thread-safe.

### Intended usage pattern (explicit capture)

```python
tracker = UsageTracker()
llm = MockLLM(responses=["hello"])

def answer(state):
    resp = llm.generate([Message(role="user", content=state["query"])])
    tracker.add(resp.usage)
    state["answer"] = resp.content
    return state
```

## Why explicit capture for now

Automatic capture requires the executor to know which LLM calls belong to a
run — i.e. a run-scoped context threaded into nodes. Building that machinery
now would change node signatures before the context design is settled, so we
ship the explicit tracker first. The public API stays stable and usage works
today; the "zero-touch" promise is Phase 3.

## Deferred to Phase 3 (context + auto-capture)

- Run-scoped `RunContext` with automatic usage capture (nodes call
  `llm.generate(...)` and the framework records usage — no manual
  `tracker.add`).
- Per-node and per-run token budgets with fail-fast enforcement.
- `on_usage` / `on_run` hooks for analytics and cost dashboards.
- Budget-aware callbacks and cancellation semantics.

## Deferred to later phases

- Response caching and cache-aware pricing.
- Model tiering / routing.
- Streaming and tool/function-calling support (extend `LLM` or add a second
  protocol when needed).
- `LLM` lifecycle hooks so the executor can reuse sessions across runs.

## Provider adapters (live outside the core)

OpenAI, Anthropic, Gemini, etc. are separate adapters that implement `LLM`
and translate `Message` <-> native payloads. They are also responsible for
computing `Usage.cost` from their pricing tables. Ship them as extras
(`pyantra[openai]`, ...) so the core has zero dependencies.

## Replay connection

`MockLLM` is the seam for deterministic replay: a recorded trace replays into
a mock that returns the recorded responses. Keep the `LLM` boundary pure — do
not import real providers inside core or in tests.
