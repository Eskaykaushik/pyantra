"""Runtime type guards for state and node I/O.

``typecheck``/``assert_type`` validate values against annotated types
(including ``Union``, ``Optional``, and container generics) at runtime.
``assert_state`` checks a state dataclass instance against its field
annotations, catching drift between declared and actual runtime types.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, get_type_hints

from pyantra import PyantraError


class TypeGuardError(PyantraError):
    """Raised when a value violates an expected runtime type."""


def _unwrap(expected: Any) -> Any:
    """Strip ``Annotated`` so its underlying type is checked."""
    if typing.get_origin(expected) is typing.Annotated:
        return typing.get_args(expected)[0]
    return expected


def typecheck(value: Any, expected: Any) -> bool:
    """Return whether ``value`` satisfies ``expected`` at runtime.

    Supports plain types, ``typing.Any``, ``Union``/``X | Y``, ``Optional``,
    ``Annotated``, and ``list``/``dict``/``set``/``tuple`` generics.
    """
    expected = _unwrap(expected)
    origin = typing.get_origin(expected)
    if origin is None:
        if expected is Any:
            return True
        return isinstance(value, expected)

    args = typing.get_args(expected)
    if origin is typing.Union or origin is types.UnionType:
        return any(typecheck(value, arg) for arg in args)

    if origin is list:
        return isinstance(value, list) and all(
            typecheck(item, args[0]) for item in value
        )
    if origin is set:
        return isinstance(value, set) and all(
            typecheck(item, args[0]) for item in value
        )
    if origin is dict:
        key_type, value_type = args
        return isinstance(value, dict) and all(
            typecheck(key, key_type) and typecheck(item, value_type)
            for key, item in value.items()
        )
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return isinstance(value, tuple) and all(
                typecheck(item, args[0]) for item in value
            )
        return isinstance(value, tuple) and len(value) == len(args) and all(
            typecheck(item, expected_item)
            for item, expected_item in zip(value, args, strict=True)
        )
    return isinstance(value, origin)


def assert_type(value: Any, expected: Any, *, name: str = "value") -> None:
    """Raise :class:`TypeGuardError` if ``value`` does not match ``expected``."""
    if not typecheck(value, expected):
        raise TypeGuardError(
            f"{name} must be {expected!r}, got {type(value).__name__}"
        )


def check_state(state: Any) -> list[str]:
    """Return the names of dataclass fields whose runtime type mismatches.

    Annotations are resolved with ``get_type_hints``; unresolvable fields
    are skipped.
    """
    if not dataclasses.is_dataclass(state):
        raise TypeGuardError("check_state requires a dataclass instance")
    try:
        hints = get_type_hints(state.__class__, include_extras=True)
    except (NameError, TypeError):
        return []
    return [
        name
        for name, expected in hints.items()
        if not typecheck(getattr(state, name), expected)
    ]


def assert_state(state: Any) -> None:
    """Raise :class:`TypeGuardError` if any state field has a mismatched type."""
    mismatches = check_state(state)
    if mismatches:
        raise TypeGuardError(
            f"State fields have unexpected types: {', '.join(mismatches)}"
        )


__all__ = [
    "TypeGuardError",
    "assert_state",
    "assert_type",
    "check_state",
    "typecheck",
]
