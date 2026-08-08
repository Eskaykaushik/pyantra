"""Shared fixtures for Pyantra tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pyantra import Graph


@dataclass
class State:
    value: int = 0
    history: list[str] = field(default_factory=list)


@pytest.fixture
def graph() -> Graph[State]:
    return Graph(State)
