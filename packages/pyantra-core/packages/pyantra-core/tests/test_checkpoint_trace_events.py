"""Tests for persisted checkpoint traces including interrupt and failure events."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pyantra import (
    Graph,
    SQLiteCheckpointStore,
    interrupt,
    RunStatus,
)


@dataclass
class S:
    value: int = 0


def test_interrupt_persists_events(tmp_path) -> None:
    db = str(tmp_path / "interrupts.db")
    store = SQLiteCheckpointStore(db)

    g = Graph(S)

    @g.node
    def start(s: S) -> S:
        return s

    @g.node
    def review(s: S) -> S:
        _ = interrupt({"question": "ok?"})
        return s

    g.set_entry_point(start)
    g.add_edge(start, review)
    app = g.compile()

    run = app.run(S(), checkpointer=store, run_id="run-int")
    assert run.status == RunStatus.PAUSED

    cp = store.load("run-int")
    assert cp is not None
    assert any(e.event == "node.interrupted" for e in cp.events)


def test_failure_persists_events(tmp_path) -> None:
    db = str(tmp_path / "failures.db")
    store = SQLiteCheckpointStore(db)

    g = Graph(S)

    @g.node
    def start(s: S) -> S:
        return s

    @g.node
    def boom(s: S) -> S:
        raise RuntimeError("boom")

    g.set_entry_point(start)
    g.add_edge(start, boom)
    app = g.compile()

    run = app.run(S(), checkpointer=store, run_id="run-fail")
    assert run.status == RunStatus.FAILED

    cp = store.load("run-fail")
    assert cp is not None
    # Expect the persisted checkpoint to include the node.failed and/or run.failed
    assert any(e.event in ("node.failed", "run.failed") for e in cp.events)
