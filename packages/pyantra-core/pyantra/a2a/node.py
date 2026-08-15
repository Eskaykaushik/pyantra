"""A2A delegation: hand a task to a remote agent from inside a graph.

:class:`DelegateNode` is the bridge between pyantra state and an A2A
``Task``. It sends the task, waits out its lifecycle (``submitted`` →
``working`` → terminal), and merges the outcome into ``result_field`` through
the graph's reducers like any other node. When the agent requests input the
node pauses the run with :func:`~pyantra.runtime.interrupt.interrupt`, so the
delegation composes with pyantra's human-in-the-loop machinery instead of
reimplementing it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, TypeAlias

from pyantra.a2a.client import A2aClientProtocol
from pyantra.a2a.errors import A2aError
from pyantra.a2a.types import Message, Task, TaskStatus, TextPart
from pyantra.graph.node import Node, NodeConfig
from pyantra.runtime.errors import GraphCompileError
from pyantra.runtime.interrupt import _run_context, interrupt
from pyantra.state.state import StateT, StateUpdate
from pyantra.tools.base import _field_names

PayloadFrom: TypeAlias = Callable[[StateT], Message | dict[str, Any]]
ResultFrom: TypeAlias = Callable[[StateT, Task], Any]

_NON_TERMINAL = (TaskStatus.SUBMITTED, TaskStatus.WORKING)
_INTERRUPT_KIND = "a2a.input-required"


class DelegateNode(Node[StateT]):
    """Delegates a task to a remote agent via A2A and merges the outcome.

    The node owns a single A2A ``Task`` lifecycle per invocation:

    * ``payload_from(state)`` builds the initial ``Message`` (defaults to the
      state's ``str`` as a text part). It may return a ``Message`` or a raw
      ``{"role", "parts"}`` dict.
    * the task is sent with ``tasks/send`` and polled (``tasks/get``) until it
      reaches a terminal state;
    * on completion the agent's reply is merged into ``result_field`` — the
      last agent message's text by default, or ``result_from(state, task)``
      when given;
    * ``FAILED``/``CANCELED``/``UNKNOWN`` tasks raise
      :class:`~pyantra.a2a.A2aError`.

    Input-required negotiation
    -------------------------

    When the agent reports ``input-required`` the run pauses via
    :func:`~pyantra.runtime.interrupt.interrupt` (this requires a
    checkpointer). The interrupt payload carries the task — including its
    ``id`` — plus the agent's last message. Resuming with ``resume(run_id,
    value, checkpointer=...)`` re-enters the node, which sends ``value`` back
    through ``message/send`` and continues polling, so multi-turn handoffs
    stay on one A2A task.

    ``task_id_field`` optionally persists the remote task id into state when
    the delegation succeeds (and between turns it can be recovered from the
    persisted interrupt payload when a checkpointer is in use).
    """

    def __init__(
        self,
        *,
        name: str,
        client: A2aClientProtocol,
        result_field: str,
        payload_from: Callable[[StateT], Message | dict[str, Any]] | None = None,
        task_id_field: str | None = None,
        result_from: Callable[[StateT, Task], Any] | None = None,
        poll_interval: float = 1.0,
        max_wait: float = 300.0,
        config: NodeConfig | None = None,
    ) -> None:
        super().__init__(name, self._invoke, config)
        self._client = client
        self._result_field = result_field
        self._payload_from = payload_from
        self._task_id_field = task_id_field
        self._result_from = result_from
        self._poll_interval = poll_interval
        self._max_wait = max_wait

    def validate(self, state_type: type[Any]) -> None:
        """Fail compilation early when the node touches unknown state fields."""
        fields = _field_names(state_type)
        if fields is not None and self._result_field not in fields:
            raise GraphCompileError(
                f"DelegateNode {self.name!r} writes to unknown state field "
                f"{self._result_field!r}."
            )
        if (
            self._task_id_field is not None
            and fields is not None
            and self._task_id_field not in fields
        ):
            raise GraphCompileError(
                f"DelegateNode {self.name!r} writes task id to unknown state "
                f"field {self._task_id_field!r}."
            )

    async def _invoke(self, state: StateT) -> StateUpdate:
        update: StateUpdate = {}
        task_id = (
            getattr(state, self._task_id_field, None) if self._task_id_field else None
        )
        ctx = _run_context.get()
        response = ctx.responses.pop(self.name, None) if ctx is not None else None

        if response is not None:
            if not task_id and ctx is not None:
                task_id = self._recover_task_id(ctx)
            if not task_id:
                raise A2aError(
                    f"Node {self.name!r} resumed with input but the remote task "
                    "id is not available; set task_id_field or keep the run "
                    "resumable."
                )
            task = await self._client.send_message(
                task_id,
                Message(role="user", parts=[TextPart(text=str(response))]),
            )
        elif task_id:
            task = await self._client.get_task(task_id)
        else:
            task = await self._client.send_task(self._build_message(state))
            if self._task_id_field is not None:
                update[self._task_id_field] = task.id

        task = await self._wait_for_terminal(task)

        if task.status is TaskStatus.INPUT_REQUIRED:
            self._persist_task_id(task.id, state, ctx)
            interrupt(self._interrupt_payload(task))
            raise AssertionError("interrupt() should have paused the run")
        if task.status is TaskStatus.UNKNOWN:
            raise A2aError(
                f"agent task {task.id!r} is in an unknown state; resend the task."
            )
        if task.status is TaskStatus.FAILED:
            raise A2aError(f"agent task {task.id!r} failed.")
        if task.status is TaskStatus.CANCELED:
            raise A2aError(f"agent task {task.id!r} was canceled.")

        update[self._result_field] = self._extract_result(state, task)
        return update

    async def _wait_for_terminal(self, task: Task) -> Task:
        """Poll until the task reaches a terminal state or ``max_wait`` elapses."""
        deadline = time.monotonic() + self._max_wait
        while task.status in _NON_TERMINAL:
            if time.monotonic() >= deadline:
                raise A2aError(
                    f"agent task {task.id!r} did not reach a terminal state "
                    f"within max_wait={self._max_wait}s."
                )
            await asyncio.sleep(self._poll_interval)
            task = await self._client.get_task(task.id)
        return task

    def _build_message(self, state: StateT) -> Message:
        if self._payload_from is not None:
            payload = self._payload_from(state)
        else:
            payload = {"role": "user", "parts": [{"kind": "text", "text": str(state)}]}
        return payload if isinstance(payload, Message) else Message.from_dict(payload)

    def _extract_result(self, state: StateT, task: Task) -> Any:
        if self._result_from is not None:
            return self._result_from(state, task)
        for message in reversed(task.messages):
            if message.role != "agent":
                continue
            texts = [part.text for part in message.parts if isinstance(part, TextPart)]
            if texts:
                return "\n".join(texts)
        return None

    def _interrupt_payload(self, task: Task) -> dict[str, Any]:
        last = task.messages[-1] if task.messages else None
        return {
            "kind": _INTERRUPT_KIND,
            "task": task.to_dict(),
            "message": last.to_dict() if last else None,
        }

    def _persist_task_id(
        self, task_id: str, state: StateT, ctx: Any
    ) -> None:
        """Make the remote task id survive an input-required pause.

        ``interrupt()`` only checkpoints (and pauses) when a checkpointer is in
        use, and partial node updates are merged only when the node returns.
        So before pausing we write the task id into the live state *and* the
        persisted checkpoint, which the executor re-reads on resume — this
        works for both object-identity stores (memory) and serializing stores
        (SQLite, DBOS).
        """
        if self._task_id_field is None:
            return
        setattr(state, self._task_id_field, task_id)
        if ctx is None or ctx.checkpointer is None:
            return
        checkpoint = ctx.checkpointer.load(ctx.run_id)
        if checkpoint is None:
            return
        setattr(checkpoint.state, self._task_id_field, task_id)
        ctx.checkpointer.save(checkpoint)

    def _recover_task_id(self, ctx: Any) -> str | None:
        """Re-derive the task id from this node's persisted interrupt payload.

        ``interrupt()`` only pauses (and checkpoints) when a checkpointer is in
        use, so on a resumed run the task id survives in
        ``checkpoint.interrupts`` even when no ``task_id_field`` is set.
        """
        if ctx.checkpointer is None:
            return None
        checkpoint = ctx.checkpointer.load(ctx.run_id)
        if checkpoint is None:
            return None
        for node, payload in reversed(checkpoint.interrupts):
            if node != self.name:
                continue
            if isinstance(payload, dict):
                task = payload.get("task")
                if isinstance(task, dict):
                    task_id = task.get("id") or task.get("taskId")
                    if isinstance(task_id, str):
                        return task_id
        return None


__all__ = ["DelegateNode", "PayloadFrom", "ResultFrom"]
