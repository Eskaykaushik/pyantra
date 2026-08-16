"""Tests for human-in-the-loop interrupts and resume."""

from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass, field
from typing import Annotated

import pytest

from pyantra import (
    Graph,
    MemoryCheckpointStore,
    NodeConfig,
    PyantraError,
    RunStatus,
    interrupt,
)


@dataclass
class ApprovalState:
    draft: str = ""
    decision: str = ""
    history: list[str] = field(default_factory=list)


@dataclass
class SkipState:
    out: Annotated[list[str], operator.add] = field(default_factory=list)
    decision: str = ""


def test_interrupt_pauses_run_with_payload() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    def draft(state: ApprovalState) -> dict[str, str]:
        return {"draft": "hi"}

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        decision = interrupt({"action": "approve", "draft": state.draft})
        return {"decision": decision}

    graph.set_entry_point(draft)
    graph.add_edge(draft, review)

    app = graph.compile()
    run = app.run(ApprovalState(), checkpointer=store, run_id="approval-1")

    assert run.status == RunStatus.PAUSED
    assert run.interrupt == {"action": "approve", "draft": "hi"}
    assert run.state.draft == "hi"
    assert store.load("approval-1") is not None

    resumed = app.resume("approval-1", "yes", checkpointer=store)

    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "yes"


def test_interrupt_requires_checkpointer() -> None:
    graph = Graph(ApprovalState)

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        decision = interrupt("approved")
        return {"decision": decision}

    graph.set_entry_point(review)

    run = graph.compile().run(ApprovalState())

    assert run.status == RunStatus.FAILED
    assert run.error is not None
    assert "checkpointer" in run.error


def test_interrupt_outside_node_raises() -> None:
    with pytest.raises(PyantraError):
        interrupt("nope")


def test_resume_without_checkpoint_raises() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        decision = interrupt("approved")
        return {"decision": decision}

    graph.set_entry_point(review)
    app = graph.compile()

    with pytest.raises(Exception, match="No checkpoint"):
        app.resume("missing", "value", checkpointer=store)


def test_interrupt_survives_retry_policy() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)
    calls = {"n": 0}

    @graph.node(config=NodeConfig(retries=1))
    def review(state: ApprovalState) -> dict[str, str]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        decision = interrupt("approve")
        return {"decision": decision}

    graph.set_entry_point(review)
    app = graph.compile()
    run_id = "retry-run"

    run = app.run(ApprovalState(), checkpointer=store, run_id=run_id)

    assert run.status == RunStatus.PAUSED
    assert run.interrupt == "approve"

    resumed = app.resume(run_id, "approved", checkpointer=store)
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "approved"


async def test_interrupt_async() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    async def review(state: ApprovalState) -> dict[str, str]:
        decision = interrupt("question")
        return {"decision": decision}

    graph.set_entry_point(review)
    app = graph.compile()

    run = await app.arun(ApprovalState(), checkpointer=store, run_id="async-run")
    assert run.status == RunStatus.PAUSED
    assert run.interrupt == "question"

    resumed = await app.aresume("async-run", "answer", checkpointer=store)
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "answer"


async def test_resume_inside_running_loop_raises() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        decision = interrupt("approve")
        return {"decision": decision}

    graph.set_entry_point(review)
    app = graph.compile()

    run = await app.arun(ApprovalState(), checkpointer=store, run_id="resume-loop")
    assert run.status == RunStatus.PAUSED

    with pytest.raises(RuntimeError, match="running event loop"):
        app.resume("resume-loop", "yes", checkpointer=store)


async def test_resume_sync_inside_running_loop() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        decision = interrupt("approve")
        return {"decision": decision}

    graph.set_entry_point(review)
    app = graph.compile()

    run = await app.arun(ApprovalState(), checkpointer=store, run_id="resume-sync")
    assert run.status == RunStatus.PAUSED

    resumed = app.resume_sync("resume-sync", "yes", checkpointer=store)

    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "yes"


def test_interrupt_events_include_node_interrupted() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        interrupt("blocked")
        return {"decision": "never"}

    graph.set_entry_point(review)
    app = graph.compile()

    run = app.run(ApprovalState(), checkpointer=store, run_id="events-run")
    events = {e.event for e in run.events}

    assert "node.interrupted" in events
    assert "run.paused" in events


def test_interrupt_not_swallowed_by_node_except_exception() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        try:
            decision = interrupt("approve")
        except Exception:
            decision = "swallowed"
        return {"decision": decision}

    graph.set_entry_point(review)
    app = graph.compile()

    run = app.run(ApprovalState(), checkpointer=store, run_id="swallow-run")

    assert run.status == RunStatus.PAUSED
    assert run.interrupt == "approve"


def test_interrupt_checkpoint_persists_payload(tmp_path) -> None:
    from pyantra import SQLiteCheckpointStore

    db = str(tmp_path / "interrupts.db")
    store: SQLiteCheckpointStore[ApprovalState] = SQLiteCheckpointStore(db)
    graph = Graph(ApprovalState)

    @graph.node
    def review(state: ApprovalState) -> dict[str, str]:
        decision = interrupt("needs-input")
        return {"decision": decision}

    graph.set_entry_point(review)
    app = graph.compile()

    run = app.run(ApprovalState(), checkpointer=store, run_id="sqlite-int")

    assert run.status == RunStatus.PAUSED
    checkpoint = store.load("sqlite-int")
    assert checkpoint is not None
    assert checkpoint.interrupts == [("review", "needs-input")]

    resumed = app.resume("sqlite-int", "provided", checkpointer=store)
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "provided"
    store.close()


def test_sequential_interrupts_at_multiple_nodes() -> None:
    store: MemoryCheckpointStore[ApprovalState] = MemoryCheckpointStore()
    graph = Graph(ApprovalState)

    @graph.node
    def first(state: ApprovalState) -> dict[str, str]:
        state.history.append(interrupt("first-question"))
        return {"history": state.history}

    @graph.node
    def second(state: ApprovalState) -> dict[str, str]:
        state.history.append(interrupt("second-question"))
        return {"history": state.history}

    graph.set_entry_point(first)
    graph.add_edge(first, second)
    app = graph.compile()
    run_id = "multi-int"

    run1 = app.run(ApprovalState(), checkpointer=store, run_id=run_id)
    assert run1.status == RunStatus.PAUSED
    assert run1.interrupt == "first-question"

    run2 = app.resume(run_id, "answer-1", checkpointer=store)
    assert run2.status == RunStatus.PAUSED
    assert run2.interrupt == "second-question"

    run3 = app.resume(run_id, "answer-2", checkpointer=store)
    assert run3.status == RunStatus.COMPLETED
    assert run3.state is not None
    assert run3.state.history == ["answer-1", "answer-2"]


def test_interrupt_inside_parallel_branch() -> None:

    @dataclass
    class ParState:
        out: Annotated[list[str], operator.add] = field(default_factory=list)
        decision: str = ""

    store: MemoryCheckpointStore[ParState] = MemoryCheckpointStore()
    graph = Graph(ParState)

    @graph.node
    def start(state: ParState) -> ParState:
        return state

    @graph.node
    def branch_a(state: ParState) -> dict[str, list[str]]:
        return {"out": ["a"]}

    @graph.node
    def branch_b(state: ParState) -> dict[str, str]:
        decision = interrupt({"q": "ok?"})
        return {"decision": decision}

    @graph.node
    def finish(state: ParState) -> ParState:
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, branch_a, branch_b, join=finish)
    app = graph.compile()

    run = app.run(ParState(), checkpointer=store, run_id="par-int")
    assert run.status == RunStatus.PAUSED
    assert run.interrupt == {"q": "ok?"}

    resumed = app.resume("par-int", "yes", checkpointer=store)
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "yes"
    assert resumed.state.out == ["a"]


async def test_interrupt_cancels_siblings_and_resume_reruns_them() -> None:
    @dataclass
    class ParState:
        out: Annotated[list[str], operator.add] = field(default_factory=list)
        decision: str = ""

    calls = {"slow": 0}
    store: MemoryCheckpointStore[ParState] = MemoryCheckpointStore()
    graph = Graph(ParState)

    @graph.node
    def start(state: ParState) -> ParState:
        return state

    @graph.node
    async def slow(state: ParState) -> dict[str, list[str]]:
        await asyncio.sleep(0.2)
        calls["slow"] += 1
        return {"out": ["slow"]}

    @graph.node
    def ask(state: ParState) -> dict[str, str]:
        decision = interrupt("go?")
        return {"decision": decision}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, slow, ask)
    app = graph.compile()

    run = await app.arun(ParState(), checkpointer=store, run_id="rerun")

    assert run.status == RunStatus.PAUSED
    assert calls["slow"] == 0, "in-flight sibling was not cancelled on interrupt"
    event_names = [e.event for e in run.events]
    assert event_names[-1] == "run.paused"
    assert "node.completed" not in event_names[
        event_names.index("run.paused") + 1 :
    ]

    resumed = await app.aresume("rerun", "yes", checkpointer=store)

    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "yes"
    assert resumed.state.out == ["slow"]
    assert calls["slow"] == 1, "cancelled sibling should run exactly once on resume"


def test_parallel_interrupt_resume_skips_completed_sibling() -> None:
    @dataclass
    class ParState:
        out: Annotated[list[str], operator.add] = field(default_factory=list)
        decision: str = ""

    calls = {"side": 0}
    store: MemoryCheckpointStore[ParState] = MemoryCheckpointStore()
    graph = Graph(ParState)

    @graph.node
    def start(state: ParState) -> ParState:
        return state

    @graph.node
    def side_effect(state: ParState) -> dict[str, list[str]]:
        calls["side"] += 1
        return {"out": ["result"]}

    @graph.node
    def ask(state: ParState) -> dict[str, str]:
        decision = interrupt("go?")
        return {"decision": decision}

    @graph.node
    def finish(state: ParState) -> ParState:
        return state

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, side_effect, ask, join=finish)
    app = graph.compile()
    run_id = "skip-completed"

    run = app.run(ParState(), checkpointer=store, run_id=run_id)

    assert run.status == RunStatus.PAUSED
    assert calls["side"] == 1
    assert run.state is not None
    assert run.state.out == ["result"], "completed sibling result was not preserved"

    checkpoint = store.load(run_id)
    assert checkpoint is not None
    assert checkpoint.parallel is not None
    assert checkpoint.parallel.completed == ("side_effect",)
    assert checkpoint.parallel.pending == ("ask",)
    assert checkpoint.parallel.interrupted == "ask"

    resumed = app.resume(run_id, "yes", checkpointer=store)

    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "yes"
    assert resumed.state.out == ["result"]
    assert calls["side"] == 1, "completed sibling re-ran on resume"
    side_completed = [
        e
        for e in resumed.events
        if e.event == "node.completed" and e.node == "side_effect"
    ]
    assert len(side_completed) == 1


def test_parallel_interrupt_resume_skips_completed_sibling_sqlite(tmp_path) -> None:
    from pyantra import SQLiteCheckpointStore

    calls = {"side": 0}
    store: SQLiteCheckpointStore[SkipState] = SQLiteCheckpointStore(
        str(tmp_path / "par-skip.db")
    )
    graph = Graph(SkipState)

    @graph.node
    def start(state: SkipState) -> SkipState:
        return state

    @graph.node
    def side_effect(state: SkipState) -> dict[str, list[str]]:
        calls["side"] += 1
        return {"out": ["result"]}

    @graph.node
    def ask(state: SkipState) -> dict[str, str]:
        decision = interrupt("go?")
        return {"decision": decision}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, side_effect, ask)
    app = graph.compile()

    run = app.run(SkipState(), checkpointer=store, run_id="sqlite-skip")

    assert run.status == RunStatus.PAUSED
    assert calls["side"] == 1

    resumed = app.resume("sqlite-skip", "yes", checkpointer=store)

    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state is not None
    assert resumed.state.decision == "yes"
    assert resumed.state.out == ["result"]
    assert calls["side"] == 1, "completed sibling re-ran on resume"
    store.close()
