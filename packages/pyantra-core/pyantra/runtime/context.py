"""Run-scoped context published to the node currently executing.

A node (or a wrapper it calls, such as a caching LLM) can read the active
:class:`RunContext` from the :data:`run_context` context variable to emit
structured events and record LLM usage. Both flow into the run's event trace
and are aggregated onto :attr:`~pyantra.runtime.run.Run.usage`, so cost and
cache behavior stay observable without a parallel logging system.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic

from pyantra.checkpoint.base import CheckpointStore
from pyantra.llm.types import Usage
from pyantra.runtime.run import Run, RunEvent
from pyantra.state.state import StateT


@dataclass
class RunContext(Generic[StateT]):
    """The context available to a node while it executes.

    The executor publishes one per node invocation and resets it when the
    node finishes. Nodes and LLM wrappers may call :meth:`emit` to append a
    :class:`~pyantra.runtime.run.RunEvent` and :meth:`record_usage` to
    accumulate LLM usage into the run's :attr:`Run.usage`.
    """

    run_id: str
    node: str
    responses: dict[str, Any]
    checkpointer: CheckpointStore[StateT] | None
    _run: Run[StateT] | None = None

    def emit(
        self,
        event: str,
        *,
        node: str | None = None,
        duration_ms: float | None = None,
        message: str | None = None,
    ) -> None:
        """Append an event to the active run's trace.

        ``node`` defaults to the node this context belongs to.
        """
        run = self._run
        if run is None:
            return
        run.events.append(
            RunEvent(
                run_id=run.run_id,
                event=event,
                timestamp=time.time(),
                node=node if node is not None else self.node,
                duration_ms=duration_ms,
                message=message,
            )
        )

    def record_usage(self, usage: Usage) -> None:
        """Accumulate ``usage`` into the run's aggregate and trace it.

        Appends a ``usage.recorded`` event carrying the usage so per-node cost
        attribution and durable resume both work off the same event stream.
        """
        run = self._run
        if run is None:
            return
        run.usage = run.usage + usage
        run.events.append(
            RunEvent(
                run_id=run.run_id,
                event="usage.recorded",
                timestamp=time.time(),
                node=self.node,
                usage=usage,
            )
        )


run_context: ContextVar[RunContext[Any] | None] = ContextVar(
    "pyantra_run_context", default=None
)


__all__ = ["RunContext", "run_context"]
