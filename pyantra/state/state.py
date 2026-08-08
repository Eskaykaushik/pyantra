"""Typed workflow state.

Any object may act as workflow state; a ``dataclass`` is recommended so that
Pyantra can validate the flow of state between nodes at runtime.
"""

from __future__ import annotations

from typing import Protocol, TypeVar


class State(Protocol):
    """Marker protocol for workflow state objects."""


StateT = TypeVar("StateT", bound=State)
