from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, Union

import pytest
from pyantra_guard import (
    TypeGuardError,
    assert_state,
    assert_type,
    check_state,
    typecheck,
)


def test_typecheck_scalars() -> None:
    assert typecheck(1, int)
    assert typecheck("x", str)
    assert not typecheck(1, str)


def test_typecheck_any() -> None:
    assert typecheck(object(), Any)
    assert typecheck(1, Any)


def test_typecheck_union_and_optional() -> None:
    assert typecheck(1, Union[int, str])  # noqa: UP007
    assert typecheck("x", Union[int, str])  # noqa: UP007
    assert not typecheck(1.0, Union[int, str])  # noqa: UP007
    assert typecheck(None, Optional[str])  # noqa: UP045
    assert typecheck("x", str | None)
    assert typecheck(None, int | None)


def test_typecheck_containers() -> None:
    assert typecheck([1, 2], list[int])
    assert not typecheck([1, "x"], list[int])
    assert typecheck({"a": 1}, dict[str, int])
    assert not typecheck({"a": "x"}, dict[str, int])
    assert typecheck({1, 2}, set[int])
    assert typecheck((1, 2), tuple[int, int])
    assert not typecheck((1, 2, 3), tuple[int, int])
    assert typecheck((1, 2, 3), tuple[int, ...])


def test_typecheck_annotated() -> None:
    assert typecheck("x", Annotated[str, "meta"])


def test_assert_type_passes_and_raises() -> None:
    assert_type(1, int)
    with pytest.raises(TypeGuardError, match="value must be"):
        assert_type("x", int)
    with pytest.raises(TypeGuardError, match="count must be"):
        assert_type("x", int, name="count")


def test_assert_state_valid() -> None:
    @dataclass
    class State:
        count: int
        labels: list[str]

    assert_state(State(count=1, labels=["a"]))


def test_check_state_reports_mismatches() -> None:
    @dataclass
    class State:
        count: int
        labels: list[str]

    assert check_state(State(count="oops", labels=[1, 2])) == ["count", "labels"]


def test_assert_state_raises_on_mismatch() -> None:
    @dataclass
    class State:
        count: int
        labels: list[str] = field(default_factory=list)

    with pytest.raises(TypeGuardError, match="count"):
        assert_state(State(count="oops"))


def test_check_state_requires_dataclass() -> None:
    with pytest.raises(TypeGuardError, match="dataclass"):
        check_state(1)
