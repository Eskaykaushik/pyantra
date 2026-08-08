"""Executor: runs a compiled graph and produces a Run object.

The executor core is asynchronous. The synchronous ``CompiledGraph.run``
bridges to it with ``asyncio.run`` so there is a single traversal
implementation shared by both sync and async execution.
"""

from __future__ import annotations

import inspect
import time
import uuid
from collections.abc import Awaitable
from typing import Generic, cast

from pyantra.graph.compiler import CompiledGraph
from pyantra.graph.node import Node
from pyantra.runtime.errors import (
    GraphExecutionError,
    InvalidRouteError,
    MaxIterationsError,
    NodeExecutionError,
)
from pyantra.runtime.run import Run, RunEvent, RunStatus
from pyantra.state.state import StateT


class Executor(Generic[StateT]):
    """Executes a compiled graph, producing a structured ``Run``."""

    def __init__(self, graph: CompiledGraph[StateT]) -> None:
        self._graph = graph

    async def arun(self, state: StateT, *, max_iterations: int = 100) -> Run[StateT]:
        run = Run[StateT](
            run_id=uuid.uuid4().hex,
            status=RunStatus.PENDING,
            state=state,
        )
        self._emit(run, "run.started")
        try:
            final_state = await self._execute(run, state, max_iterations)
            run.state = final_state
            run.status = RunStatus.COMPLETED
            self._emit(run, "run.completed")
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.exception = exc
            self._emit(run, "run.failed", message=run.error)
        return run

    async def _execute(
        self,
        run: Run[StateT],
        state: StateT,
        max_iterations: int,
    ) -> StateT:
        if not isinstance(state, self._graph.state_type):
            raise GraphExecutionError(
                f"Expected state of type {self._graph.state_type.__name__!r}, "
                f"got {type(state).__name__!r}.",
                run_id=run.run_id,
            )

        current: str | None = self._graph.entry_point
        iterations = 0
        while current is not None:
            if iterations >= max_iterations:
                raise MaxIterationsError(
                    f"Run exceeded max_iterations={max_iterations} at node "
                    f"{current!r}; possible non-terminating loop.",
                    run_id=run.run_id,
                )
            iterations += 1

            node = self._graph.nodes[current]
            self._emit(run, "node.started", node=current)
            started = time.perf_counter()
            try:
                result = await _invoke(node, state)
            except Exception as exc:
                duration_ms = _duration_ms(started)
                self._emit(
                    run,
                    "node.failed",
                    node=current,
                    duration_ms=duration_ms,
                    message=str(exc),
                )
                raise NodeExecutionError(
                    f"Node {current!r} failed: {exc}",
                    run_id=run.run_id,
                    node=current,
                ) from exc

            if result is not None and not isinstance(result, self._graph.state_type):
                raise NodeExecutionError(
                    f"Node {current!r} returned {type(result).__name__!r}, expected "
                    f"{self._graph.state_type.__name__!r}.",
                    run_id=run.run_id,
                    node=current,
                )
            if result is not None:
                state = result

            self._emit(
                run,
                "node.completed",
                node=current,
                duration_ms=_duration_ms(started),
            )
            run.state = state
            current = await self._next(run, current, state)
        return state

    async def _next(self, run: Run[StateT], current: str, state: StateT) -> str | None:
        conditional = self._graph.conditional_edges.get(current, [])
        if conditional:
            edge = conditional[0]
            key = await _resolve_key(edge.router(state))
            if edge.path_map is not None:
                target = edge.path_map.get(key)
                if target is None:
                    target = edge.default
                if target is None or target not in self._graph.nodes:
                    raise InvalidRouteError(
                        f"Router for node {current!r} returned unknown route {key!r}.",
                        run_id=run.run_id,
                    )
            else:
                target = key
                if target not in self._graph.nodes:
                    raise InvalidRouteError(
                        f"Router for node {current!r} returned unknown node "
                        f"{target!r}.",
                        run_id=run.run_id,
                    )
            self._emit(run, "edge.selected", node=current, message=target)
            return target

        edges = self._graph.edges.get(current, [])
        if not edges:
            self._emit(run, "edge.selected", node=current, message="END")
            return None
        target = edges[0].target
        self._emit(run, "edge.selected", node=current, message=target or "END")
        return target

    def _emit(
        self,
        run: Run[StateT],
        event: str,
        *,
        node: str | None = None,
        duration_ms: float | None = None,
        message: str | None = None,
    ) -> None:
        run.events.append(
            RunEvent(
                run_id=run.run_id,
                event=event,
                timestamp=time.time(),
                node=node,
                duration_ms=duration_ms,
                message=message,
            )
        )


async def _invoke(node: Node[StateT], state: StateT) -> StateT | None:
    result = node(state)
    if inspect.isawaitable(result):
        result = await cast(Awaitable[StateT | None], result)
    return result


async def _resolve_key(raw: str | Awaitable[str]) -> str:
    if inspect.isawaitable(raw):
        return await raw
    return raw


def _duration_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
