"""Executor: runs a compiled graph and produces a Run object.

The executor core is asynchronous. The synchronous ``CompiledGraph.run``
bridges to it with ``asyncio.run`` so there is a single traversal
implementation shared by both sync and async execution.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
import uuid
from collections.abc import Awaitable
from typing import Generic, cast

from pyantra.checkpoint.base import Checkpoint, CheckpointStore
from pyantra.graph.compiler import CompiledGraph
from pyantra.graph.node import Node, NodeConfig
from pyantra.graph.parallel import ParallelEdge
from pyantra.llm.types import Usage
from pyantra.reliability.retry import compute_delay, is_retryable
from pyantra.reliability.timeout import with_timeout
from pyantra.runtime.context import RunContext, run_context
from pyantra.runtime.errors import (
    GraphExecutionError,
    InvalidRouteError,
    MaxIterationsError,
    NodeExecutionError,
    NodeTimeoutError,
    RetryExhaustedError,
)
from pyantra.runtime.interrupt import GraphInterrupt
from pyantra.runtime.run import Run, RunEvent, RunStatus
from pyantra.state.reducers import apply_updates, diff_state, merge_state
from pyantra.state.state import StateT, StateUpdate


class Executor(Generic[StateT]):
    """Executes a compiled graph, producing a structured ``Run``."""

    def __init__(self, graph: CompiledGraph[StateT]) -> None:
        self._graph = graph
        self._checkpointer: CheckpointStore[StateT] | None = None
        self._interrupt_responses: dict[str, object] = {}

    async def arun(
        self,
        state: StateT,
        *,
        max_iterations: int = 100,
        checkpointer: CheckpointStore[StateT] | None = None,
        run_id: str | None = None,
        interrupt_responses: dict[str, object] | None = None,
    ) -> Run[StateT]:
        run_id = run_id or uuid.uuid4().hex
        self._checkpointer = checkpointer
        self._interrupt_responses = dict(interrupt_responses or {})

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
        run.usage = _sum_usage(run.events)
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
        except GraphInterrupt as exc:
            run.status = RunStatus.PAUSED
            run.interrupt = exc.payload
            self._emit(run, "run.paused", message=str(exc.payload))
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
            result = await self._run_node(run, node, state)
            if result is not None:
                state = self._merge_result(run, node.name, state, result)
            run.state = state
            current = await self._next(run, current, state)
        return state

    async def _run_node(
        self,
        run: Run[StateT],
        node: Node[StateT],
        state: StateT,
    ) -> StateT | StateUpdate | None:
        """Execute a single node with reliability policy and emit events.

        Returns the node's raw result (state, partial update, or ``None``);
        merging is the caller's responsibility so parallel branches can merge
        into a shared state.
        """
        self._emit(run, "node.started", node=node.name)
        started = time.perf_counter()
        token = run_context.set(
            RunContext[StateT](
                run_id=run.run_id,
                node=node.name,
                responses=self._interrupt_responses,
                checkpointer=self._checkpointer,
                _run=run,
            )
        )
        try:
            result = await self._invoke_with_policy(run, node, state)
        except GraphInterrupt as exc:
            duration_ms = _duration_ms(started)
            self._emit(
                run,
                "node.interrupted",
                node=node.name,
                duration_ms=duration_ms,
                message=str(exc.payload),
            )
            self._persist_interrupt(run, node.name, exc.payload)
            raise
        except Exception as exc:
            duration_ms = _duration_ms(started)
            self._emit(
                run,
                "node.failed",
                node=node.name,
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
                exc.node = node.name
            raise
        finally:
            run_context.reset(token)
        self._emit(
            run,
            "node.completed",
            node=node.name,
            duration_ms=_duration_ms(started),
        )
        return result

    def _persist_interrupt(
        self,
        run: Run[StateT],
        node_name: str,
        payload: object,
    ) -> None:
        """Record a pending interruption in the checkpoint for durability."""
        checkpointer = self._checkpointer
        if checkpointer is None:
            return
        checkpoint = checkpointer.load(run.run_id)
        if checkpoint is None:
            return
        checkpoint.interrupts.append((node_name, payload))
        checkpointer.save(checkpoint)

    async def _invoke_with_policy(
        self,
        run: Run[StateT],
        node: Node[StateT],
        state: StateT,
    ) -> StateT | StateUpdate | None:
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
            except (asyncio.TimeoutError, TimeoutError) as exc:
                last_exc = TimeoutError(str(exc))
                self._emit(
                    run,
                    "node.attempt.timeout",
                    node=node.name,
                    message=f"Timeout after {config.timeout}s",
                )
                if not _should_retry(config, last_exc):
                    break
            except Exception as exc:
                last_exc = exc
                self._emit(
                    run,
                    "node.attempt.failed",
                    node=node.name,
                    message=str(exc),
                )
                if not _should_retry(config, exc):
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

        if isinstance(last_exc, TimeoutError):
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

    async def _invoke_node(
        self, node: Node[StateT], state: StateT
    ) -> StateT | StateUpdate | None:
        result = node(state)
        if inspect.isawaitable(result):
            return await cast(Awaitable[StateT | StateUpdate | None], result)
        return result

    def _merge_result(
        self,
        run: Run[StateT],
        node_name: str,
        state: StateT,
        result: StateT | StateUpdate,
    ) -> StateT:
        """Merge a node result into ``state`` in place and return ``state``.

        * ``dict`` — partial updates applied per-field (reducers run).
        * the state type — merged field by field (reducers run); if the node
          returned the very object it received, the mutation is already in
          place and nothing is merged.
        """
        try:
            if isinstance(result, dict):
                return apply_updates(state, result, self._graph.reducers)
            if isinstance(result, self._graph.state_type):
                return merge_state(state, result, self._graph.reducers)
        except Exception as exc:
            raise NodeExecutionError(
                f"Node {node_name!r} returned state that failed to merge: {exc}",
                run_id=run.run_id,
                node=node_name,
            ) from exc
        raise NodeExecutionError(
            f"Node {node_name!r} returned {type(result).__name__!r}, expected "
            f"{self._graph.state_type.__name__!r} or a partial update dict.",
            run_id=run.run_id,
            node=node_name,
        )

    def _checkpoint(
        self,
        checkpointer: CheckpointStore[StateT],
        run: Run[StateT],
        resume_at: str,
        state: StateT,
    ) -> None:
        # Preserve pending interruptions: a resumed run re-reads the checkpoint
        # while its node executes, so wiping the history here would lose the
        # interrupt that prompted the resume.
        pending: list[tuple[str, object]] = []
        existing = checkpointer.load(run.run_id)
        if existing is not None:
            pending = list(existing.interrupts)
        checkpointer.save(
            Checkpoint(
                run_id=run.run_id,
                resume_at=resume_at,
                state=state,
                events=list(run.events),
                interrupts=pending,
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

        parallel = self._graph.parallel_edges.get(current)
        if parallel is not None:
            return await self._run_parallel(run, parallel[0], state)

        edges = self._graph.edges.get(current, [])
        if not edges:
            self._emit(run, "edge.selected", node=current, message="END")
            return None
        target = edges[0].target
        self._emit(run, "edge.selected", node=current, message=target or "END")
        return target

    async def _run_parallel(
        self,
        run: Run[StateT],
        parallel: ParallelEdge,
        state: StateT,
    ) -> str | None:
        """Run all parallel targets concurrently and merge their results.

        Returns the join node name, or ``None`` to end the workflow. A branch
        that returns ``None`` contributes nothing.

        Each branch executes on a deep copy of ``state``, so a branch that
        mutates its copy and returns it already contains the pre-existing
        content. Every branch result is therefore diffed against ``state``
        (the pristine pre-fan-out snapshot) *before* any merge runs, so only
        the branch's additions flow through the field reducers and pre-existing
        reducer state is never re-applied once per branch.
        """
        self._emit(
            run,
            "edge.selected",
            node=parallel.source,
            message=" || ".join(parallel.targets),
        )
        results = await asyncio.gather(
            *(self._run_branch(run, target, state) for target in parallel.targets)
        )
        updates: list[tuple[str, StateUpdate]] = []
        for branch_name, result in results:
            if result is None:
                continue
            if isinstance(result, self._graph.state_type):
                result = diff_state(state, result, self._graph.reducers)
            updates.append((branch_name, cast(StateUpdate, result)))
        merged = state
        for branch_name, update in updates:
            merged = self._merge_result(run, branch_name, merged, update)
        run.state = merged
        if parallel.join is not None:
            self._emit(
                run, "edge.selected", node=parallel.source, message=parallel.join
            )
            return parallel.join
        self._emit(run, "edge.selected", node=parallel.source, message="END")
        return None

    async def _run_branch(
        self,
        run: Run[StateT],
        target: str,
        state: StateT,
    ) -> tuple[str, StateT | StateUpdate | None]:
        """Execute one parallel branch on an isolated copy of the state."""
        node = self._graph.nodes[target]
        branch_state = copy.deepcopy(state)
        result = await self._run_node(run, node, branch_state)
        return target, result

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


def _should_retry(config: NodeConfig, exc: BaseException) -> bool:
    """Whether a failure should be retried under ``config``.

    Never-retryable exceptions (``@non_retryable``) always fail immediately.
    When ``config.retry_on`` is set, only failures matching one of the listed
    exception types are retried; everything else fails immediately.
    """
    if not is_retryable(exc):
        return False
    if config.retry_on is not None:
        return isinstance(exc, config.retry_on)
    return True


def _duration_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _sum_usage(events: list[RunEvent]) -> Usage:
    """Aggregate the usage carried by a run's events (e.g. after a resume)."""
    total = Usage()
    for event in events:
        if event.usage is not None:
            total = total + event.usage
    return total
