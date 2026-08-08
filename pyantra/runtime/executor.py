"""Executor: runs a compiled graph and produces a Run object.

The executor core is asynchronous. The synchronous ``CompiledGraph.run``
bridges to it with ``asyncio.run`` so there is a single traversal
implementation shared by both sync and async execution.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable
from typing import Generic, cast

from pyantra.checkpoint.base import Checkpoint, CheckpointStore
from pyantra.graph.compiler import CompiledGraph
from pyantra.graph.node import Node
from pyantra.reliability.retry import compute_delay, is_retryable
from pyantra.reliability.timeout import with_timeout
from pyantra.runtime.errors import (
    GraphExecutionError,
    InvalidRouteError,
    MaxIterationsError,
    NodeExecutionError,
    NodeTimeoutError,
    RetryExhaustedError,
)
from pyantra.runtime.run import Run, RunEvent, RunStatus
from pyantra.state.state import StateT


class Executor(Generic[StateT]):
    """Executes a compiled graph, producing a structured ``Run``."""

    def __init__(self, graph: CompiledGraph[StateT]) -> None:
        self._graph = graph

    async def arun(
        self,
        state: StateT,
        *,
        max_iterations: int = 100,
        checkpointer: CheckpointStore[StateT] | None = None,
        run_id: str | None = None,
    ) -> Run[StateT]:
        run_id = run_id or uuid.uuid4().hex

        resume_at: str | None = None
        events: list[RunEvent] = []
        if checkpointer is not None:
            checkpoint = checkpointer.load(run_id)
            if checkpoint is not None:
                state = checkpoint.state
                events = list(checkpoint.events)
                resume_at = checkpoint.resume_at

        run = Run[StateT](
            run_id=run_id,
            status=RunStatus.PENDING,
            state=state,
            events=events,
        )
        self._emit(run, "run.resumed" if resume_at else "run.started")
        try:
            final_state = await self._execute(
                run, state, max_iterations, checkpointer, resume_at
            )
            run.state = final_state
            run.status = RunStatus.COMPLETED
            self._emit(run, "run.completed")
            if checkpointer is not None:
                checkpointer.delete(run.run_id)
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
        checkpointer: CheckpointStore[StateT] | None,
        resume_at: str | None,
    ) -> StateT:
        if not isinstance(state, self._graph.state_type):
            raise GraphExecutionError(
                f"Expected state of type {self._graph.state_type.__name__!r}, "
                f"got {type(state).__name__!r}.",
                run_id=run.run_id,
            )

        current: str | None = resume_at or self._graph.entry_point
        iterations = 0
        while current is not None:
            if iterations >= max_iterations:
                raise MaxIterationsError(
                    f"Run exceeded max_iterations={max_iterations} at node "
                    f"{current!r}; possible non-terminating loop.",
                    run_id=run.run_id,
                )
            iterations += 1

            if checkpointer is not None:
                self._checkpoint(checkpointer, run, current, state)

            node = self._graph.nodes[current]
            self._emit(run, "node.started", node=current)
            started = time.perf_counter()
            try:
                result = await self._invoke_with_policy(run, node, state)
            except Exception as exc:
                duration_ms = _duration_ms(started)
                self._emit(
                    run,
                    "node.failed",
                    node=current,
                    duration_ms=duration_ms,
                    message=str(exc),
                )
                if isinstance(exc, GraphExecutionError) and exc.run_id is None:
                    exc.run_id = run.run_id
                if (
                    isinstance(
                        exc,
                        (NodeExecutionError, NodeTimeoutError, RetryExhaustedError),
                    )
                    and exc.node is None
                ):
                    exc.node = current
                raise

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

    async def _invoke_with_policy(
        self,
        run: Run[StateT],
        node: Node[StateT],
        state: StateT,
    ) -> StateT | None:
        config = node.config
        breaker = config.breaker
        if breaker is not None:
            breaker.before(node=node.name, run_id=run.run_id)

        attempts = config.retries + 1
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                if config.timeout is not None:
                    result = await with_timeout(
                        self._invoke_node(node, state), config.timeout
                    )
                else:
                    result = await self._invoke_node(node, state)
            except asyncio.TimeoutError as exc:
                last_exc = exc
                self._emit(
                    run,
                    "node.attempt.timeout",
                    node=node.name,
                    message=f"Timeout after {config.timeout}s",
                )
            except Exception as exc:
                last_exc = exc
                self._emit(
                    run,
                    "node.attempt.failed",
                    node=node.name,
                    message=str(exc),
                )
                if not is_retryable(exc):
                    break
            else:
                if breaker is not None:
                    breaker.record_success()
                return result

            if attempt < attempts:
                delay = compute_delay(
                    config.backoff, attempt, config.base_delay, config.max_delay
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                self._emit(
                    run,
                    "node.retrying",
                    node=node.name,
                    message=f"Attempt {attempt + 1}/{attempts}",
                )

        if breaker is not None:
            breaker.record_failure()

        if isinstance(last_exc, asyncio.TimeoutError):
            raise NodeTimeoutError(
                f"Node {node.name!r} exceeded timeout of {config.timeout}s.",
                run_id=run.run_id,
                node=node.name,
            ) from last_exc

        assert last_exc is not None
        if attempt >= attempts and config.retries > 0:
            raise RetryExhaustedError(
                f"Node {node.name!r} failed after {attempts} attempts: {last_exc}",
                run_id=run.run_id,
                node=node.name,
            ) from last_exc
        raise NodeExecutionError(
            f"Node {node.name!r} failed: {last_exc}",
            run_id=run.run_id,
            node=node.name,
        ) from last_exc

    async def _invoke_node(self, node: Node[StateT], state: StateT) -> StateT | None:
        result = node(state)
        if inspect.isawaitable(result):
            result = await cast(Awaitable[StateT | None], result)
        return result

    def _checkpoint(
        self,
        checkpointer: CheckpointStore[StateT],
        run: Run[StateT],
        resume_at: str,
        state: StateT,
    ) -> None:
        checkpointer.save(
            Checkpoint(
                run_id=run.run_id,
                resume_at=resume_at,
                state=state,
                events=list(run.events),
            )
        )

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


async def _resolve_key(raw: str | Awaitable[str]) -> str:
    if inspect.isawaitable(raw):
        return await raw
    return raw


def _duration_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
