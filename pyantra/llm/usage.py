"""Per-run LLM usage and cost tracking."""

from __future__ import annotations

import threading

from pyantra.llm.types import Usage


class UsageTracker:
    """Accumulates usage across model invocations.

    A tracker is created by the caller (typically one per run) and handed to
    nodes that make model calls, or captured via closure by a shared LLM
    wrapper. Nodes record usage explicitly for now; automatic per-run capture
    is deferred to Phase 3 (see ``docs/llm.md``).
    """

    def __init__(self) -> None:
        self._total = Usage()
        self._lock = threading.Lock()

    @property
    def total(self) -> Usage:
        """Aggregate usage across all recorded calls so far."""
        with self._lock:
            return self._total

    def add(self, usage: Usage) -> None:
        """Record a single call's usage into the running total."""
        with self._lock:
            self._total = self._total + usage

    def reset(self) -> None:
        """Clear all recorded usage (e.g. to reuse a tracker per run)."""
        with self._lock:
            self._total = Usage()


__all__ = ["UsageTracker"]
