"""State merging: reducers and partial updates.

State fields may be annotated with a reducer to change how updates combine
instead of overwriting::

    from typing import Annotated, operator

    @dataclass
    class State:
        value: int
        messages: Annotated[list[str], operator.add]

    @graph.node
    def append(state: State) -> dict[str, list[str]]:
        return {"messages": ["hi"]}

Nodes may return:

* ``None`` — the node mutated state in place (reducers are not applied).
* the state type — the returned object is merged field by field; annotated
  fields are reduced against the current values, other fields replace.
* a ``dict`` of field updates — the same merge, applied per key.

Reducers are extracted from ``typing.Annotated`` metadata at compile time. Any
callable ``(current, update) -> new`` works; ``operator.add`` on lists is the
canonical example.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, TypeVar, get_type_hints

Reducer: TypeAlias = Callable[[Any, Any], Any]
S = TypeVar("S")


def add(current: list[Any], update: Any) -> list[Any]:
    """Append ``update`` to ``current`` (like ``operator.add``)."""
    return list(current) + list(update)


def merge_dicts(current: dict[Any, Any], update: dict[Any, Any]) -> dict[Any, Any]:
    """Merge ``update`` into ``current`` key by key."""
    merged = dict(current)
    merged.update(update)
    return merged


def extract_reducers(state_type: type[Any]) -> dict[str, Reducer]:
    """Return ``{field_name: reducer}`` for fields annotated with a reducer.

    A field is ``Annotated[base, metadata, ...]``; the first callable metadata
    entry is treated as its reducer. Non-dataclass state types have no fields
    and therefore no reducers.
    """
    if not dataclasses.is_dataclass(state_type):
        return {}
    reducers: dict[str, Reducer] = {}
    try:
        hints = get_type_hints(state_type, include_extras=True)
    except Exception:
        return {}
    for name, hint in hints.items():
        metadata = getattr(hint, "__metadata__", None)
        if not metadata:
            continue
        for meta in metadata:
            if callable(meta):
                reducers[name] = meta
                break
    return reducers


def apply_updates(
    state: S,
    updates: dict[str, Any],
    reducers: Mapping[str, Reducer] | None = None,
) -> S:
    """Merge ``updates`` into ``state`` in place and return ``state``.

    Unknown field names raise ``KeyError``.
    """
    reducers = reducers or {}
    known = _field_names(state)
    for name, value in updates.items():
        if known is not None and name not in known:
            raise KeyError(
                f"Unknown state field {name!r}; valid fields: "
                f"{', '.join(sorted(known))}."
            )
        reducer = reducers.get(name)
        current = getattr(state, name)
        setattr(state, name, reducer(current, value) if reducer else value)
    return state


def merge_state(
    state: S,
    returned: Any,
    reducers: Mapping[str, Reducer] | None = None,
) -> S:
    """Merge ``returned`` into ``state`` in place and return ``state``.

    Annotated fields are reduced against the current values; all other fields
    are replaced by the returned values. When ``returned is state`` the node
    mutated state in place, so nothing is merged.
    """
    if returned is state:
        return state
    reducers = reducers or {}
    for field in dataclasses.fields(returned):
        name = field.name
        value = getattr(returned, name)
        reducer = reducers.get(name)
        current = getattr(state, name)
        setattr(state, name, reducer(current, value) if reducer else value)
    return state


def _field_names(state: Any) -> frozenset[str] | None:
    """Field names of a dataclass instance, or None for other types."""
    if dataclasses.is_dataclass(state):
        return frozenset(f.name for f in dataclasses.fields(state))
    return None


__all__ = [
    "Reducer",
    "add",
    "apply_updates",
    "extract_reducers",
    "merge_dicts",
    "merge_state",
]
