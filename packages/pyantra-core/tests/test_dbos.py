"""Tests for the DBOS-backed checkpoint store."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from pyantra import (
    END,
    DBOSCheckpointStore,
    Graph,
    MemoryCheckpointStore,
    RunStatus,
    interrupt,
)
from pyantra.checkpoint import Checkpoint, CheckpointStore
from pyantra.checkpoint.dbos import _create_datasource, _import_sqlalchemy
from pyantra.runtime.errors import PyantraError


@dataclass
class DbState:
    items: list[str]
    done: bool = False


def test_module_does_not_import_dbos_or_sqlalchemy() -> None:
    code = (
        "import sys\n"
        "import pyantra.checkpoint.dbos\n"
        "assert 'dbos' not in sys.modules, 'dbos imported at import time'\n"
        "assert 'sqlalchemy' not in sys.modules, 'sqlalchemy imported at import time'\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_import_errors_are_helpful(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    with pytest.raises(PyantraError, match=r"pyantra\[dbos\]"):
        _import_sqlalchemy()

    monkeypatch.setitem(sys.modules, "dbos", None)
    monkeypatch.setitem(sys.modules, "sqlalchemy", object())
    with pytest.raises(PyantraError, match=r"pyantra\[dbos\]"):
        _create_datasource("sqlite:///:memory:")


def test_requires_exactly_one_of_datasource_or_url() -> None:
    with pytest.raises(ValueError, match="exactly one of datasource or url"):
        DBOSCheckpointStore()
    with pytest.raises(ValueError, match="exactly one of datasource or url"):
        DBOSCheckpointStore(datasource=object(), url="sqlite:///:memory:")


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'checkpoints.db'}"


pytest.importorskip("dbos")


@pytest.mark.parametrize("ctor", ["url", "datasource"])
def test_round_trip_update_and_delete(ctor: str, sqlite_url: str) -> None:
    if ctor == "url":
        store: CheckpointStore[DbState] = DBOSCheckpointStore(url=sqlite_url)
    else:
        datasource = _create_datasource(sqlite_url)
        store = DBOSCheckpointStore(datasource=datasource)

    first = Checkpoint(
        run_id="run-1", resume_at="resume_node", state=DbState(items=["a"])
    )
    store.save(first)
    loaded = store.load("run-1")
    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.resume_at == "resume_node"
    assert loaded.state == DbState(items=["a"])

    updated = Checkpoint(
        run_id="run-1",
        resume_at="later_node",
        state=DbState(items=["a", "b"], done=True),
    )
    store.save(updated)
    reloaded = store.load("run-1")
    assert reloaded is not None
    assert reloaded.resume_at == "later_node"
    assert reloaded.state == DbState(items=["a", "b"], done=True)

    assert store.load("missing-run") is None

    store.delete("run-1")
    assert store.load("run-1") is None


def test_custom_table_name(sqlite_url: str) -> None:
    store = DBOSCheckpointStore(url=sqlite_url, table="my_checkpoints")
    checkpoint = Checkpoint(run_id="r", resume_at=None, state=DbState(items=[]))
    store.save(checkpoint)
    assert store.load("r") is not None


def test_real_run_and_resume(sqlite_url: str) -> None:
    store = DBOSCheckpointStore(url=sqlite_url)

    graph = Graph(DbState)

    @graph.node
    def ask(state: DbState) -> dict:
        response = interrupt("what next?")
        return {"items": state.items + [response]}

    graph.set_entry_point(ask)
    graph.add_edge(ask, END)

    app = graph.compile()
    first = app.run(DbState(items=["start"]), checkpointer=store, run_id="dbos-run")

    assert first.status == RunStatus.PAUSED
    assert first.interrupt == "what next?"

    resumed = app.resume("dbos-run", "answer", checkpointer=store)
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state == DbState(items=["start", "answer"], done=False)


def test_dbos_store_coexists_with_memory_store(sqlite_url: str) -> None:
    dbos_store = DBOSCheckpointStore(url=sqlite_url)
    memory_store: MemoryCheckpointStore[DbState] = MemoryCheckpointStore()

    checkpoint = Checkpoint(run_id="shared", resume_at=None, state=DbState(items=["x"]))
    dbos_store.save(checkpoint)
    memory_store.save(checkpoint)

    assert dbos_store.load("shared") == memory_store.load("shared")
