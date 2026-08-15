"""Checkpoint serialization.

Durable checkpoint stores need to write :class:`Checkpoint` objects to disk.
The serializer is pluggable: :class:`JsonSerializer` is the safe, portable
default (no arbitrary code runs on load), and :class:`PickleSerializer`
supports arbitrary object graphs at the cost of safety — never load pickle
checkpoints from an untrusted source.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import pickle
import types
import typing
from abc import ABC, abstractmethod
from typing import Any, Generic, cast, get_type_hints

from pyantra.checkpoint.base import Checkpoint
from pyantra.state.state import StateT


class Serializer(ABC, Generic[StateT]):
    """Serializes and deserializes whole :class:`Checkpoint` objects."""

    @abstractmethod
    def dumps(self, checkpoint: Checkpoint[StateT]) -> bytes:
        """Serialize ``checkpoint`` to bytes for storage."""

    @abstractmethod
    def loads(self, data: bytes) -> Checkpoint[StateT]:
        """Deserialize bytes previously produced by :meth:`dumps`."""


def _jsonify(value: Any) -> Any:
    """Convert tuples/sets to lists so values survive ``json`` round-trips."""
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    return value


def _unwrap(expected: Any) -> Any:
    """Strip ``Annotated`` metadata from a type hint."""
    if typing.get_origin(expected) is typing.Annotated:
        return typing.get_args(expected)[0]
    return expected


def _dataclass_origin(expected: Any) -> type | None:
    """Return the dataclass type a hint describes, or None.

    Handles bare dataclass types, ``Optional[X]`` / ``X | None`` with a single
    dataclass, and lists/dicts are handled by the caller.
    """
    expected = _unwrap(expected)
    origin = typing.get_origin(expected)
    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in typing.get_args(expected) if arg is not type(None)]
        if len(args) == 1:
            expected = args[0]
            origin = typing.get_origin(expected)
    target = origin or expected
    if isinstance(target, type) and dataclasses.is_dataclass(target):
        return target
    return None


def _import_type(module: str, qualname: str) -> type:
    """Resolve a ``module.qualname`` dotted path to a class."""
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise ValueError(f"Checkpoint state type {qualname!r} is not a class.")
    return obj


def _coerce(expected: Any, value: Any) -> Any:
    """Rebuild nested dataclass values inside a JSON-decoded structure."""
    if expected is None or value is None:
        return value
    expected = _unwrap(expected)
    origin = typing.get_origin(expected)
    if isinstance(value, list) and origin is list:
        args = typing.get_args(expected)
        item = _dataclass_origin(args[0]) if args else None
        if item is not None:
            return [_reconstruct(item, entry) for entry in value]
        return value
    if isinstance(value, dict):
        item = _dataclass_origin(expected)
        if item is not None:
            return _reconstruct(item, value)
        if origin is dict:
            args = typing.get_args(expected)
            value_type = _dataclass_origin(args[1]) if len(args) == 2 else None
            if value_type is not None:
                return {
                    key: _reconstruct(value_type, entry) for key, entry in value.items()
                }
    return value


def _reconstruct(cls: type[Any], fields: dict[str, Any]) -> Any:
    """Rebuild a dataclass instance from a JSON-decoded field dict."""
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {}
    return cls(
        **{name: _coerce(hints.get(name), value) for name, value in fields.items()}
    )


class JsonSerializer(Serializer[StateT]):
    """JSON-backed checkpoint serialization for dataclass states.

    Safe by default: nothing beyond ``json.loads`` and the state class's
    ``__init__`` runs when loading, so checkpoints cannot trigger arbitrary
    code execution. State fields and interrupt payloads must be
    JSON-serializable — primitives, ``list``/``tuple``/``set``/``dict``, and
    nested dataclasses. Use :class:`PickleSerializer` for arbitrary objects.
    """

    def dumps(self, checkpoint: Checkpoint[StateT]) -> bytes:
        body = {
            "run_id": checkpoint.run_id,
            "resume_at": checkpoint.resume_at,
            "state": self._encode_state(checkpoint.state),
            "events": [event.to_dict() for event in checkpoint.events],
            "interrupts": [[node, payload] for node, payload in checkpoint.interrupts],
        }
        try:
            return json.dumps(body).encode("utf-8")
        except TypeError as exc:
            raise TypeError(
                f"Checkpoint is not JSON-serializable ({exc}); use "
                "PickleSerializer for arbitrary object graphs."
            ) from exc

    def loads(self, data: bytes) -> Checkpoint[StateT]:
        body = json.loads(data.decode("utf-8"))
        return Checkpoint(
            run_id=body["run_id"],
            resume_at=body["resume_at"],
            state=self._decode_state(body["state"]),
            events=[_reconstruct_event(event) for event in body["events"]],
            interrupts=[(node, payload) for node, payload in body["interrupts"]],
        )

    def _encode_state(self, state: StateT) -> dict[str, Any]:
        if not dataclasses.is_dataclass(state):
            raise TypeError(
                "JsonSerializer requires a dataclass state; use "
                "PickleSerializer for other state types."
            )
        cls = type(state)
        return {
            "__type__": [cls.__module__, cls.__qualname__],
            "fields": _jsonify(dataclasses.asdict(cast(Any, state))),
        }

    def _decode_state(self, data: dict[str, Any]) -> StateT:
        module, qualname = data["__type__"]
        cls = _import_type(module, qualname)
        if not dataclasses.is_dataclass(cls):
            raise ValueError(
                f"Checkpoint state type {qualname!r} is not a dataclass."
            )
        return cast(StateT, _reconstruct(cls, data["fields"]))


class PickleSerializer(Serializer[StateT]):
    """Pickle-backed serialization for arbitrary object graphs.

    Do not use with checkpoints from untrusted sources: unpickling arbitrary
    bytes can execute code. Prefer :class:`JsonSerializer`.
    """

    def dumps(self, checkpoint: Checkpoint[StateT]) -> bytes:
        return pickle.dumps(checkpoint)

    def loads(self, data: bytes) -> Checkpoint[StateT]:
        return cast(Checkpoint[StateT], pickle.loads(data))


def _reconstruct_event(data: dict[str, Any]) -> Any:
    from pyantra.runtime.run import RunEvent

    return RunEvent(**data)


__all__ = ["JsonSerializer", "PickleSerializer", "Serializer"]
