 
# Pyantra — CLAUDE.md

## Project Overview

Pyantra is an open-source Python framework for building production-grade AI agents and multi-step LLM workflows.

The core philosophy is:

> **Agents should be observable, reproducible, testable, reliable, and cost-aware by default.**

Pyantra is library-first and self-hosted. It is not a model provider, SaaS platform, or no-code agent builder.

---

# Primary Goals

Pyantra should make it possible to:

1. Build typed, stateful LLM workflows.
2. Execute workflows as graphs of nodes and edges.
3. Reliably retry and recover failed nodes.
4. Inspect exactly what happened during a run.
5. Replay previous runs deterministically.
6. Control context and token usage.
7. Support multi-agent delegation.
8. Pause workflows for human approval.
9. Regression-test agent behavior.
10. Run with minimal infrastructure and dependencies.

---

# v1 Non-Goals

Do NOT attempt to build:

* A hosted SaaS platform.
* A UI/no-code workflow builder.
* A model provider.
* A fine-tuning framework.
* Full LangGraph feature parity.
* A large plugin ecosystem.
* A distributed execution engine.

Prefer a small, composable core.

---

# Design Principles

## 1. Library First

Pyantra must work entirely as a Python library.

A basic workflow should require nothing more than:

```python
from pyantra import Graph
```

Avoid requiring databases, Redis, Docker, cloud services, or web servers for basic usage.

---

## 2. Typed by Default

Use Python type hints throughout the codebase.

Prefer:

```python
from dataclasses import dataclass


@dataclass
class AgentState:
    query: str
    result: str | None = None
```

over untyped dictionaries where practical.

For user-defined state, support standard Python typing and validation.

---

## 3. Explicit State

Every workflow has state.

Nodes receive state and produce state changes.

Conceptually:

```text
State
  ↓
Node
  ↓
State Update
  ↓
Next Node
```

Avoid hidden global state.

---

# Initial Architecture

Start with the following package structure:

```text
pyantra/
├── __init__.py
│
├── graph/
│   ├── __init__.py
│   ├── graph.py
│   ├── node.py
│   ├── edge.py
│   ├── conditional.py
│   └── compiler.py
│
├── runtime/
│   ├── __init__.py
│   ├── executor.py
│   ├── run.py
│   └── errors.py
│
├── state/
│   ├── __init__.py
│   └── state.py
│
├── reliability/
│   ├── __init__.py
│   ├── retry.py
│   ├── timeout.py
│   └── circuit_breaker.py
│
├── checkpoint/
│   ├── __init__.py
│   ├── base.py
│   └── memory.py
│
├── tracing/
│   ├── __init__.py
│   ├── events.py
│   └── tracer.py
│
├── context/
│   ├── __init__.py
│   ├── manager.py
│   ├── budget.py
│   └── compression.py
│
├── llm/
│   ├── __init__.py
│   ├── base.py
│   └── mock.py
│
├── agents/
│   ├── __init__.py
│   └── handoff.py
│
├── human/
│   ├── __init__.py
│   └── approval.py
│
├── evaluation/
│   ├── __init__.py
│   └── regression.py
│
└── storage/
    ├── __init__.py
    └── base.py
```

Do not implement all modules immediately.

Build them incrementally according to the roadmap below.

---

# Core API Philosophy

The public API should be small.

A user should eventually be able to write:

```python
from dataclasses import dataclass

from pyantra import Graph


@dataclass
class State:
    question: str
    answer: str | None = None


graph = Graph(State)


@graph.node
def answer_question(state: State):
    state.answer = "42"
    return state


graph.set_entry_point(answer_question)

app = graph.compile()

result = app.run(
    State(question="What is the answer?")
)

print(result.state.answer)
```

The exact API may evolve.

Prioritize:

* readability
* discoverability
* type safety
* minimal boilerplate

---

# Graph Model

The fundamental abstraction is:

```text
Graph
 ├── Nodes
 ├── Edges
 ├── Conditional Edges
 └── State
```

A node should have a clear execution boundary.

Example:

```python
@graph.node
def retrieve(state: State) -> State:
    ...
```

Edges define execution order.

```python
graph.add_edge(retrieve, generate)
```

Conditional routing should eventually support:

```python
graph.add_conditional_edges(
    classify,
    route
)
```

---

# Compiler

`Graph.compile()` should validate the graph before execution.

Initial compiler responsibilities:

* Verify an entry point exists.
* Verify referenced nodes exist.
* Detect unreachable nodes.
* Detect invalid edges.
* Validate conditional routes.
* Detect obvious invalid cycles.
* Validate termination configuration.

Compilation should fail early.

Prefer errors such as:

```text
GraphCompileError:
Node 'generate_report' is unreachable from entry point 'start'.
```

over cryptic runtime failures.

---

# Runtime

The runtime executes a compiled graph.

Responsibilities:

* Execute nodes.
* Resolve edges.
* Maintain state.
* Generate execution events.
* Handle failures.
* Apply retry policies.
* Check iteration limits.
* Create checkpoints.
* Support resume.

Conceptually:

```text
Run
 │
 ├── Node Started
 ├── Node Completed
 ├── Edge Selected
 ├── Node Started
 ├── Node Failed
 ├── Retry
 └── Run Completed
```

---

# Run Object

Every execution should produce a run object.

Example:

```python
result = app.run(state)

result.run_id
result.state
result.status
result.events
```

Possible statuses:

```python
PENDING
RUNNING
COMPLETED
FAILED
PAUSED
CANCELLED
```

Use enums rather than arbitrary strings.

---

# Reliability

Reliability is a first-class feature.

Each node should eventually support configuration such as:

```python
NodeConfig(
    retries=3,
    backoff="exponential",
    timeout=30,
)
```

Initial retry strategy:

```text
attempt 1
   ↓
failure
   ↓
wait
   ↓
attempt 2
   ↓
failure
   ↓
wait
   ↓
attempt 3
```

Do not retry errors that are explicitly marked non-retryable.

---

# Checkpointing

Checkpointing allows:

```text
Node A ✓
Node B ✓
Node C ✗
```

to resume from:

```text
Node C
```

rather than:

```text
Node A
```

Initial implementation:

```text
MemoryCheckpointStore
```

Later:

```text
SQLiteCheckpointStore
PostgresCheckpointStore
RedisCheckpointStore
```

Define an abstract storage interface before implementing external databases.

---

# Tracing

Every important runtime transition must produce a structured event.

Example:

```python
{
    "run_id": "...",
    "node": "retrieve",
    "event": "node.completed",
    "timestamp": "...",
    "duration_ms": 120,
}
```

Do not make logs the primary observability mechanism.

Structured events should be the source of truth.

Potential events:

```text
run.started
run.completed
run.failed

node.started
node.completed
node.failed
node.retrying

edge.selected

checkpoint.created
checkpoint.restored

human.approval_requested
human.approval_received

agent.handoff
```

---

# Deterministic Replay

Pyantra must eventually support replaying a previous execution.

Example:

```python
app.replay(run_id)
```

During replay, external LLM/model responses should be replaceable with recorded or mocked responses.

The goal is:

```text
Production failure
       ↓
Stored trace
       ↓
Replay
       ↓
Reproduce failure locally
       ↓
Debug
       ↓
Fix
       ↓
Regression test
```

This is one of Pyantra's most important differentiators.

---

# Context Management

Do not pass the entire conversation/history into every node.

Instead use scoped context:

```text
Global State
     │
     ├── Node A Context
     │
     ├── Node B Context
     │
     └── Node C Context
```

Each node should eventually be able to define:

```python
context_budget=4000
```

Context manager responsibilities:

* Select relevant context.
* Maintain context versions.
* Compress/summarize context.
* Enforce token budgets.
* Isolate agent contexts.
* Track context used during execution.

---

# Token Optimization

Pyantra should treat token usage as a runtime concern.

Support eventually:

## Zero-token routing

Logic-only operations should not invoke an LLM.

Example:

```python
if state.score > 0.8:
    return "approve"
```

No model call should occur.

## Model tiering

Allow:

```text
cheap model
    ↓
simple classification

strong model
    ↓
complex reasoning
```

## Response caching

Identical or cacheable requests should be able to reuse previous responses.

## Budgets

Support:

```text
per-node token budget
per-run token budget
per-run cost budget
```

Every model invocation should record usage when available.

---

# LLM Abstraction

Do not tightly couple Pyantra to OpenAI, Anthropic, Gemini, Azure, etc.

Define a minimal provider interface.

Example:

```python
class LLM:
    async def generate(...):
        ...
```

Model providers should be adapters.

Pyantra owns orchestration, not model APIs.

---

# Multi-Agent Architecture

Agents should be composable as workflow nodes.

A future API may look like:

```python
research_agent.delegate(
    analysis_agent,
    context={"task": "analyze findings"}
)
```

Handoffs must be:

* explicit
* scoped
* observable
* traceable
* permission-aware

Avoid uncontrolled agent-to-agent communication.

---

# Human-in-the-Loop

Human approval should be a native runtime primitive.

Conceptually:

```text
Agent
  ↓
Approval Required
  ↓
PAUSED
  ↓
Human edits/approves
  ↓
RESUMED
  ↓
Next Node
```

The runtime must preserve state while paused.

Example future API:

```python
approval = human.approval(
    message="Approve this transaction?"
)
```

---

# Testing

Pyantra should make workflows testable without real LLM calls.

Provide mocks:

```python
MockLLM(...)
```

Tests should be able to define deterministic responses.

Example:

```python
mock_llm.responses = [
    "response 1",
    "response 2",
]
```

---

# Regression Testing

Eventually support trace-based tests.

Example:

```python
test = TraceTest(
    name="customer_support_flow",
    trace="fixtures/customer_support.json",
)
```

Tests should detect:

* changed routing
* unexpected node execution
* increased token usage
* changed outputs
* increased latency
* failed assertions

---

# Evaluation Hooks

Nodes should eventually support evaluation callbacks.

Example:

```python
@graph.node(
    evaluators=[quality_evaluator]
)
def generate_answer(state):
    ...
```

Evaluation must not be tightly coupled to a specific evaluation provider.

---

# Storage

All persistent storage must use interfaces.

Example:

```python
class CheckpointStore(Protocol):
    def save(...):
        ...

    def load(...):
        ...

    def delete(...):
        ...
```

Initial implementation:

```text
InMemory
```

Then:

```text
SQLite
Postgres
Redis
```

Do not introduce these dependencies into the core package unless necessary.

---

# Dependencies

Keep the core dependency footprint extremely small.

Prefer Python standard library wherever reasonable.

Potential optional dependencies should be installed through extras:

```bash
pip install pyantra[sqlite]
pip install pyantra[postgres]
pip install pyantra[redis]
```

Do not force users to install every integration.

---

# Python Support

Target:

```text
Python >= 3.10
```

Use full type hints.

Use modern Python typing where appropriate.

---

# Async

AI workflows are naturally I/O-heavy.

Design the runtime to support async execution.

Primary implementation should eventually support:

```python
await app.arun(...)
```

while providing:

```python
app.run(...)
```

for synchronous workflows.

Do not duplicate the entire runtime implementation unnecessarily.

---

# Error Handling

Create explicit Pyantra exceptions.

Example:

```text
PyantraError
├── GraphCompileError
├── GraphExecutionError
├── NodeExecutionError
├── RetryExhaustedError
├── CheckpointError
├── ContextError
└── ReplayError
```

Errors should contain useful structured information.

Avoid swallowing exceptions.

---

# Project Quality

Every feature must include tests.

Minimum expectations:

```text
feature
  ├── implementation
  ├── unit tests
  └── documentation/example
```

Do not implement large features without tests.

---

# Testing Stack

Use:

```text
pytest
pytest-asyncio
```

Add property-based testing later if useful.

Test both:

* synchronous execution
* asynchronous execution

where applicable.

---

# Formatting and Linting

Use:

```text
ruff
mypy
pytest
```

Prefer Ruff for formatting/linting rather than maintaining multiple formatting tools.

Code should pass:

```bash
ruff check .
ruff format --check .
mypy .
pytest
```

before considering a feature complete.

---

# Documentation

Documentation should be example-first.

A new user should be able to go from:

```text
pip install pyantra
```

to:

```python
graph = Graph(...)
app = graph.compile()
result = app.run(...)
```

in under 10 minutes.

Prefer small examples over long conceptual explanations.

---

# Development Roadmap

Implement in this exact order.

## Phase 1 — Core Execution

Build:

* State abstraction
* Node
* Edge
* Graph
* Graph compiler
* Executor
* Run object
* Basic conditional routing
* Sync execution
* Async execution
* Basic tests

Goal:

```text
Typed graph → compile → execute → result
```

---

## Phase 2 — Reliability

Add:

* Retry policies
* Exponential backoff
* Timeouts
* Max iterations
* Circuit breaker
* Checkpoints
* Resume
* Structured execution events

Goal:

```text
Failure → recovery → resume
```

---

## Phase 3 — Context

Add:

* Context manager
* Scoped context
* Context budgets
* Context versioning
* Compression/summarization interface
* Relevance retrieval interface
* Agent context isolation

---

## Phase 4 — Token Optimization

Add:

* LLM usage tracking
* Token budgets
* Cost budgets
* Response caching
* Model routing/tiering
* Zero-token conditional routing
* Usage analytics

---

## Phase 5 — Multi-Agent + HITL

Add:

* Agent abstraction
* Agent delegation
* Scoped handoffs
* Human approval
* Pause/resume
* Approval state persistence

---

## Phase 6 — Testing and Evaluation

Add:

* Mock LLM
* Deterministic replay
* Trace recording
* Trace regression tests
* Evaluation hooks
* Golden traces
* Prompt/model version tracking

---

# Interoperability

Do not design Pyantra as an isolated ecosystem.

Eventually investigate interoperability with:

* LangGraph
* LangChain
* OpenTelemetry
* OpenAI-compatible APIs
* Anthropic-compatible APIs

An interop layer may be added later.

Do not allow interoperability requirements to complicate the core runtime prematurely.

---

# Performance

Pyantra should have minimal overhead compared to the actual LLM/network calls.

Avoid:

* unnecessary serialization
* excessive object copying
* unnecessary database calls
* synchronous blocking I/O inside async execution
* expensive tracing operations on the critical path

Benchmark runtime overhead separately from model latency.

---

# Security

Production workflows may contain sensitive data.

Never:

* log API keys
* log credentials
* automatically persist secrets
* expose environment variables through traces

Tracing should eventually support configurable redaction.

---

# Observability Philosophy

Pyantra should answer:

> "What exactly happened?"

For every run we should eventually be able to determine:

```text
What node ran?
Why did it run?
What state did it receive?
What context did it receive?
What model was called?
What prompt/version was used?
How many tokens were consumed?
How much did it cost?
What response was returned?
Why was the next node selected?
Did a retry happen?
What checkpoint was created?
```

This is a core product principle.

---

# Coding Rules for Claude

When implementing Pyantra:

1. Do not build multiple phases at once.
2. Do not prematurely implement integrations.
3. Keep public APIs small.
4. Write tests before or alongside implementation.
5. Prefer interfaces/protocols for replaceable infrastructure.
6. Avoid unnecessary dependencies.
7. Preserve backward compatibility once a public API is introduced.
8. Do not silently change public APIs.
9. Do not add features simply because another framework has them.
10. Optimize for debuggability and correctness before performance.
11. Every architectural decision should have a reason.
12. Keep modules small and cohesive.
13. Avoid circular dependencies.
14. Never hide network/model calls inside seemingly pure utilities.
15. Make failures explicit and observable.

---

# Git Workflow

Use small commits.

Preferred format:

```text
feat: add typed graph execution
feat: add conditional edges
test: add graph compiler tests
feat: add retry policy
fix: prevent unreachable node execution
```

Do not create huge commits containing unrelated changes.

---

# First Milestone

The first milestone is intentionally small.

Implement only:

```text
Graph
Node
Edge
State
Compiler
Executor
Run
```

The following must work:

```python
from dataclasses import dataclass

from pyantra import Graph


@dataclass
class State:
    value: int


graph = Graph(State)


@graph.node
def increment(state: State):
    state.value += 1
    return state


@graph.node
def double(state: State):
    state.value *= 2
    return state


graph.set_entry_point(increment)
graph.add_edge(increment, double)

app = graph.compile()

result = app.run(State(value=1))

assert result.state.value == 4
```

Also support a conditional workflow:

```text
START
  ↓
classify
  ├── positive → process_positive
  └── negative → process_negative
```

---

# First Task for Claude Code

Do NOT implement the entire roadmap.

Start by inspecting the repository.

Then:

1. Create the Python package structure.
2. Set up `pyproject.toml`.
3. Configure Python 3.10+.
4. Configure Ruff.
5. Configure mypy.
6. Configure pytest.
7. Implement the Phase 1 core primitives.
8. Write unit tests.
9. Add one end-to-end example.
10. Run the complete test suite.
11. Report what was implemented and what remains.

Before introducing a new abstraction, explain why it is necessary.

Do not ask for permission for routine implementation decisions.

If an architectural decision could significantly affect the public API, stop and explain the trade-off before implementing it.

---

# Definition of Done for Phase 1

Phase 1 is complete only when:

* [ ] Graphs can be created.
* [ ] Nodes can be registered.
* [ ] Edges can be defined.
* [ ] Conditional edges work.
* [ ] Typed state flows between nodes.
* [ ] Graphs can be compiled.
* [ ] Unreachable nodes are detected.
* [ ] Invalid graph definitions fail at compile time.
* [ ] Graphs execute synchronously.
* [ ] Graphs execute asynchronously.
* [ ] Runs have unique IDs.
* [ ] Execution errors are structured.
* [ ] Unit tests cover core behavior.
* [ ] An end-to-end example exists.
* [ ] Ruff passes.
* [ ] Mypy passes.
* [ ] Pytest passes.
* [ ] Public API is documented.

---

# Core Philosophy

Pyantra should not try to be:

> "Another agent framework with 100 features."

It should become:

> **The reliable execution layer for production AI agents.**

The differentiator is not simply graph execution.

The differentiator is:

```text
Agent
  ↓
Reliable Execution
  ↓
Observable Trace
  ↓
Deterministic Replay
  ↓
Regression Test
  ↓
Controlled Cost
  ↓
Production
```

Every architectural decision should move Pyantra toward this goal.
