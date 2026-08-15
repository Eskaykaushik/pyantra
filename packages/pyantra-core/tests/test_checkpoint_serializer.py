"""Tests for checkpoint serializers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pyantra import (
    Checkpoint,
    JsonSerializer,
    PickleSerializer,
    RunEvent,
    SQLiteCheckpointStore,
)


@dataclass
class Metadata:
    tags: list[str]
    note: str


@dataclass
class NestedState:
    value: int
    meta: Metadata | None = None
    items: list[Metadata] = field(default_factory=list)


@dataclass
class OpaqueState:
    payload: Any


def _checkpoint(state: Any) -> Checkpoint[Any]:
    return Checkpoint(
        run_id="run-1",
        resume_at="review",
        state=state,
        events=[
            RunEvent(run_id="run-1", event="node.started", timestamp=1.0, node="review")
        ],
        interrupts=[("review", {"question": "ok?"})],
    )


def test_json_roundtrips_nested_dataclass_state() -> None:
    original = _checkpoint(
        NestedState(
            value=1,
            meta=Metadata(tags=["a"], note="hi"),
            items=[Metadata(tags=["b"], note="yo")],
        )
    )

    data = JsonSerializer().dumps(original)
    loaded = JsonSerializer().loads(data)

    assert loaded.run_id == "run-1"
    assert loaded.resume_at == "review"
    assert loaded.state == original.state
    assert isinstance(loaded.state.meta, Metadata)
    assert isinstance(loaded.state.items[0], Metadata)
    assert loaded.events == original.events
    assert loaded.interrupts == [("review", {"question": "ok?"})]


def test_json_roundtrips_optional_metadata() -> None:
    original = _checkpoint(NestedState(value=1, meta=None))

    loaded = JsonSerializer().loads(JsonSerializer().dumps(original))

    assert loaded.state == original.state
    assert loaded.state.meta is None


def test_json_rejects_non_dataclass_state() -> None:
    checkpoint = _checkpoint("not a dataclass")

    with pytest.raises(TypeError, match="PickleSerializer"):
        JsonSerializer().dumps(checkpoint)


def test_json_rejects_non_serializable_fields() -> None:
    checkpoint = _checkpoint(OpaqueState(payload=object()))

    with pytest.raises(TypeError, match="PickleSerializer"):
        JsonSerializer().dumps(checkpoint)


def test_pickle_roundtrips_arbitrary_objects() -> None:
    original = _checkpoint(OpaqueState(payload={"s": {1, 2, 3}, "o": object()}))

    loaded = PickleSerializer().loads(PickleSerializer().dumps(original))

    assert type(loaded.state) is OpaqueState
    assert loaded.state.payload["s"] == {1, 2, 3}
    assert isinstance(loaded.state.payload["o"], object)
    assert loaded.events == original.events


def test_sqlite_json_roundtrips_across_instances(tmp_path) -> None:
    db = str(tmp_path / "nested.db")
    store1: SQLiteCheckpointStore[NestedState] = SQLiteCheckpointStore(db)
    store1.save(_checkpoint(NestedState(value=1, meta=Metadata(tags=["a"], note="hi"))))
    store1.close()

    store2: SQLiteCheckpointStore[NestedState] = SQLiteCheckpointStore(db)
    loaded = store2.load("run-1")
    store2.close()

    assert loaded is not None
    assert loaded.state.value == 1
    assert isinstance(loaded.state.meta, Metadata)
    assert loaded.state.meta.note == "hi"
    assert loaded.interrupts == [("review", {"question": "ok?"})]


def test_sqlite_pickle_roundtrips_across_instances(tmp_path) -> None:
    db = str(tmp_path / "opaque.db")
    store1: SQLiteCheckpointStore[OpaqueState] = SQLiteCheckpointStore(
        db, serializer=PickleSerializer()
    )
    store1.save(_checkpoint(OpaqueState(payload={"s": {1, 2, 3}})))
    store1.close()

    store2: SQLiteCheckpointStore[OpaqueState] = SQLiteCheckpointStore(
        db, serializer=PickleSerializer()
    )
    loaded = store2.load("run-1")
    store2.close()

    assert loaded is not None
    assert loaded.state.payload == {"s": {1, 2, 3}}
