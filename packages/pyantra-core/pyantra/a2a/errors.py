"""A2A-specific errors."""

from __future__ import annotations

from pyantra.runtime.errors import PyantraError


class A2aError(PyantraError):
    """Raised when an A2A request fails or an agent task ends in error.

    Covers transport failures (HTTP/JSON-RPC errors) and agent-side outcomes
    that are not negotiable — ``FAILED``, ``CANCELED``, or ``UNKNOWN`` tasks.
    """


__all__ = ["A2aError"]
