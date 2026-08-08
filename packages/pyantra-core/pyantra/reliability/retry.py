"""Retry policies and backoff strategies."""

from __future__ import annotations

from enum import Enum

from pyantra.runtime.errors import NonRetryableError


class Backoff(str, Enum):
    """Backoff strategy between retry attempts."""

    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"


def compute_delay(
    backoff: Backoff,
    attempt: int,
    base_delay: float,
    max_delay: float | None = None,
) -> float:
    """Delay (seconds) to wait before ``attempt`` (1-based) retry.

    ``Backoff.NONE`` and ``base_delay <= 0`` both produce no delay.
    """
    if backoff == Backoff.NONE or base_delay <= 0:
        return 0.0
    delay = (
        base_delay if backoff == Backoff.FIXED else base_delay * (2 ** (attempt - 1))
    )
    if max_delay is not None:
        delay = min(delay, max_delay)
    return delay


def is_retryable(exc: BaseException) -> bool:
    """Whether an exception should be retried.

    ``NonRetryableError`` and exceptions whose class or instance sets
    ``__retryable__ = False`` are never retried.
    """
    if isinstance(exc, NonRetryableError):
        return False
    return getattr(exc, "__retryable__", True) is not False


def non_retryable(exc_type: type[Exception]) -> type[Exception]:
    """Mark an exception class as non-retryable.

    Example::

        @non_retryable
        class SchemaError(Exception): ...
    """
    exc_type.__retryable__ = False  # type: ignore[attr-defined]
    return exc_type
