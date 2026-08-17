# pyantra-otel Implementation Plan

> **Important: Execute one file per step. Verify after each step before moving to the next.**

## Overview

Implement `pyantra-otel` — OpenTelemetry integration for Pyantra. Converts
`Run.events` into OTel spans, wraps LLMs with traced calls, and exports
usage as OTel metrics.

**Approach:** Short steps — one file at a time, verify after each.

## Package Structure

```
packages/pyantra-otel/
├── pyproject.toml
├── py.typed
├── README.md
├── pyantra_otel/
│   ├── __init__.py
│   ├── exporters.py       # setup_tracing() — configures TracerProvider + exporters
│   ├── instrument.py      # instrument() — post-exec event→span converter
│   ├── llm.py             # TracedLLM — wraps LLM with span per generate()
│   └── metrics.py         # UsageMetrics — token counters, cost histograms
└── tests/
    ├── __init__.py
    ├── test_instrument.py
    ├── test_llm.py
    └── test_metrics.py
```

## Design Principles

- **Post-execution conversion** — read `Run.events` after execution, no executor changes
- **Optional deps** — `opentelemetry-api` and `opentelemetry-sdk` are runtime deps
- **ContextVar integration** — read `run_context` for run_id/node propagation
- **Decorator pattern** — `TracedLLM` wraps any `LLM` implementation

## Interface Definitions

### setup_tracing (exporters.py)

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter

def setup_tracing(
    service_name: str = "pyantra",
    span_exporter: SpanExporter | None = None,
    metric_exporter: MetricExporter | None = None,
) -> TracerProvider: ...
```

### instrument (instrument.py)

```python
from pyantra import CompiledGraph, Run
from opentelemetry.trace import Tracer

def instrument(
    run: Run,
    graph: CompiledGraph | None = None,
    tracer: Tracer | None = None,
) -> None:
    """Convert a completed Run's events into OTel spans."""

def events_to_spans(
    events: list[RunEvent],
    tracer: Tracer | None = None,
) -> None:
    """Export a list of RunEvents as OTel spans."""
```

### TracedLLM (llm.py)

```python
from pyantra.llm.base import LLM
from opentelemetry.trace import Tracer

class TracedLLM(LLM):
    def __init__(self, llm: LLM, tracer: Tracer | None = None) -> None: ...
    # Delegates generate/agenerate to inner LLM, wrapping each call in a span
    # Span attributes: model, input_tokens, output_tokens, cost, duration_ms
```

### UsageMetrics (metrics.py)

```python
from pyantra import Usage
from opentelemetry.metrics import Meter

class UsageMetrics:
    def __init__(self, meter: Meter | None = None) -> None: ...
    def record(self, usage: Usage, *, run_id: str, node: str) -> None: ...
    # Tracks: pyantra.tokens.input (counter), pyantra.tokens.output (counter),
    #         pyantra.cost.usd (histogram)
```

## Span Mapping

| Event | OTel Span | Key Attributes |
|---|---|---|
| `run.started` | Root span start | `pyantra.run_id` |
| `run.completed` | Root span end | `pyantra.status`, `pyantra.total_cost` |
| `run.failed` | Root span end (ERROR) | `error.message`, `error.type` |
| `node.started` | Child span start | `pyantra.node`, `pyantra.run_id` |
| `node.completed` | Child span end | `duration_ms`, usage attrs |
| `node.failed` | Child span end (ERROR) | `error.message` |
| `node.retrying` | Span event | `attempt`, `max_retries` |
| `edge.selected` | Span event | `target` |
| `usage.recorded` | Span attributes | `tokens.input/output/cache`, `cost`, `model` |

## Dependencies

```toml
[project]
dependencies = [
    "pyantra>=0.5.1",
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
]
```

## Execution Steps

| # | Step | Files | Verify |
|---|---|---|---|
| 1 | Update pyproject.toml | `pyproject.toml` | `uv sync` |
| 2 | Add py.typed marker | `py.typed` | — |
| 3 | Create exporters.py | `exporters.py` | mypy |
| 4 | Create instrument.py | `instrument.py` | pytest |
| 5 | Create llm.py | `llm.py` | pytest |
| 6 | Create metrics.py | `metrics.py` | pytest |
| 7 | Create __init__.py | `__init__.py` | mypy |
| 8 | Create tests/__init__.py + test_instrument.py | tests | pytest |
| 9 | Create test_llm.py | tests | pytest |
| 10 | Create test_metrics.py | tests | pytest |
| 11 | Final lint + typecheck | — | ruff + mypy |

## Status

- [ ] Step 1: pyproject.toml
- [ ] Step 2: py.typed
- [ ] Step 3: exporters.py
- [ ] Step 4: instrument.py
- [ ] Step 5: llm.py
- [ ] Step 6: metrics.py
- [ ] Step 7: __init__.py
- [ ] Step 8: test_instrument.py
- [ ] Step 9: test_llm.py
- [ ] Step 10: test_metrics.py
- [ ] Step 11: Final lint + typecheck
