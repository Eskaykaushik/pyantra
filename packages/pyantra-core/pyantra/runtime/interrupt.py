"""Human-in-the-loop interruption support.

A node calls :func:`interrupt` to pause a run and request input. The run is
saved to the checkpointer with status ``PAUSED`` and the payload surfaced on
``run.interrupt``; the caller later resumes with
:meth:`~pyantra.graph.compiler.CompiledGraph.resume`.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pyantra.checkpoint.base import CheckpointStore
from pyantra.runtime.errors import PyantraError


class GraphInterrupt(BaseException):
    """Internal signal that a node requested human input.

    Deliberately derives from ``BaseException`` so user ``except Exception``
    blocks cannot swallow it.
    """

    def __init__(self, payload: Any) -> None:
        super().__init__(payload)
        self.payload = payload


@dataclass
class _RunContext:
    """State the executor publishes so ``interrupt()`` can act on it."""

    run_id: str
    node: str
    responses: dict[str, Any]
    checkpointer: CheckpointStore[Any] | None


_run_context: ContextVar[_RunContext | None] = ContextVar(
    "pyantra_run_context", default=None
)


def interrupt(payload: Any) -> Any:
    """Pause the current run and request input from a human.

    Call from inside a node. The run pauses with ``RunStatus.PAUSED`` and the
    payload becomes available on ``run.interrupt``. Resume with
    ``app.resume(run_id, value, checkpointer=...)``; at that point
    ``interrupt()`` returns ``value`` to the node.

    Requires a checkpointer so the run can be resumed later.
    """
    ctx = _run_context.get()
    if ctx is None:
        raise PyantraError("interrupt() can only be called inside a running node.")
    if ctx.node in ctx.responses:
        return ctx.responses.pop(ctx.node)
    if ctx.checkpointer is None:
        raise PyantraError(
            "interrupt() requires a checkpointer so the run can be resumed. "
            "Pass checkpointer=... to run()."
        )
    raise GraphInterrupt(payload)


__all__ = ["GraphInterrupt", "interrupt"]
