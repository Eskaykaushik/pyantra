# pyantra-studio Implementation Plan

> **Important: Execute one file per step. Verify after each step before moving to the next.**

## Overview

Implement `pyantra-studio` — a local development server and visual web debugger
for Pyantra. Browse run history, inspect event timelines, view state diffs,
and watch live execution via WebSocket.

**Approach:** Short steps — one file at a time, verify after each.

## Package Structure

```
packages/pyantra-studio/
├── pyproject.toml
├── py.typed
├── README.md
├── pyantra_studio/
│   ├── __init__.py
│   ├── store.py            # RunStore ABC + InMemoryRunStore
│   ├── api.py              # REST endpoints — /api/runs, /api/runs/{id}, /api/graph
│   ├── ws.py               # WebSocket handler — streams RunEvents in real-time
│   ├── server.py           # create_app() — wires routes, static files, WebSocket
│   ├── cli.py              # `pyantra-studio` CLI — launches server
│   └── ui/
│       ├── index.html      # SPA shell
│       ├── style.css       # Minimal styles
│       └── app.js          # Graph rendering + timeline + state diff
└── tests/
    ├── __init__.py
    ├── test_store.py
    └── test_api.py
```

## Design Principles

- **Zero frontend build** — vanilla HTML/JS/CSS, no bundler needed
- **In-memory by default** — RunStore keeps runs for the session, no DB required
- **Protocol-based** — RunStore is an ABC, users can swap in persistent backends
- **Composable** — `create_app()` returns a Starlette app, embeddable in any ASGI server

## Interface Definitions

### RunStore (store.py)

```python
class RunStore(ABC):
    async def save(self, run: Run) -> None: ...
    async def get(self, run_id: str) -> Run | None: ...
    async def list_runs(self, limit: int = 50) -> list[Run]: ...
    async def clear(self) -> None: ...

class InMemoryRunStore(RunStore):
    """In-memory store backed by a dict. Good for development."""
```

### REST API (api.py)

| Method | Path | Response |
|---|---|---|
| GET | `/api/runs` | `[{run_id, status, started_at, cost}]` |
| GET | `/api/runs/{run_id}` | Full `Run` dict with events |
| GET | `/api/graph` | Graph topology `{nodes, edges, conditional_edges}` |

### WebSocket (ws.py)

```python
async def ws_handler(websocket: WebSocket) -> None:
    """Accept connection, stream RunEvent dicts as JSON."""
```

Events pushed: `{"type": "event", "data": {run_id, event, timestamp, node, ...}}`

### create_app (server.py)

```python
def create_app(
    store: RunStore | None = None,
    graph: CompiledGraph | None = None,
    static_dir: Path | None = None,
) -> Starlette: ...
```

### CLI (cli.py)

```
$ pyantra-studio [--host 127.0.0.1] [--port 8765]
Studio running at http://127.0.0.1:8765
```

## Dependencies

```toml
[project]
dependencies = [
    "pyantra>=0.5.1",
    "starlette>=0.37",
    "uvicorn>=0.27",
    "anyio>=4.0",
]

[project.optional-dependencies]
dev = ["httpx>=0.27", "pytest>=7.4", "pytest-asyncio>=0.23"]

[project.scripts]
pyantra-studio = "pyantra_studio.cli:main"
```

## Execution Steps

| # | Step | Files | Verify |
|---|---|---|---|
| 1 | Update pyproject.toml | `pyproject.toml` | `uv sync` |
| 2 | Add py.typed marker | `py.typed` | — |
| 3 | Create store.py | `store.py` | pytest |
| 4 | Create api.py | `api.py` | pytest |
| 5 | Create ws.py | `ws.py` | pytest |
| 6 | Create server.py | `server.py` | pytest |
| 7 | Create cli.py | `cli.py` | manual |
| 8 | Create ui/index.html | `ui/index.html` | — |
| 9 | Create ui/style.css | `ui/style.css` | — |
| 10 | Create ui/app.js | `ui/app.js` | manual |
| 11 | Create __init__.py | `__init__.py` | mypy |
| 12 | Create tests/__init__.py + test_store.py | tests | pytest |
| 13 | Create test_api.py | tests | pytest |
| 14 | Final lint + typecheck | — | ruff + mypy |

## Status

- [ ] Step 1: pyproject.toml
- [ ] Step 2: py.typed
- [ ] Step 3: store.py
- [ ] Step 4: api.py
- [ ] Step 5: ws.py
- [ ] Step 6: server.py
- [ ] Step 7: cli.py
- [ ] Step 8: ui/index.html
- [ ] Step 9: ui/style.css
- [ ] Step 10: ui/app.js
- [ ] Step 11: __init__.py
- [ ] Step 12: test_store.py
- [ ] Step 13: test_api.py
- [ ] Step 14: Final lint + typecheck
