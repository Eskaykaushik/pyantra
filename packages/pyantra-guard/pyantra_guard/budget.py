"""Budget caps for LLM usage.

``Budget`` declares caps on tokens and cost; ``BudgetTracker`` accumulates
usage across a run (wrapping ``pyantra.UsageTracker``) and raises
``BudgetError`` the moment a cap is exceeded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from pyantra import PyantraError, Usage, UsageTracker


class BudgetError(PyantraError):
    """Raised when a usage budget cap is exceeded."""


@dataclass(frozen=True)
class Budget:
    """Token and cost caps for a run.

    Every field defaults to ``None``, meaning unlimited. A single record can
    violate a cap on its own; aggregate usage is what ``BudgetTracker``
    enforces.
    """

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost: float | None = None

    def exceeded_by(self, usage: Usage) -> list[str]:
        """Return descriptions of the caps ``usage`` violates (empty if none)."""
        violations: list[str] = []
        if (
            self.max_input_tokens is not None
            and usage.input_tokens > self.max_input_tokens
        ):
            violations.append(
                f"input tokens {usage.input_tokens} > {self.max_input_tokens}"
            )
        if (
            self.max_output_tokens is not None
            and usage.output_tokens > self.max_output_tokens
        ):
            violations.append(
                f"output tokens {usage.output_tokens} > {self.max_output_tokens}"
            )
        if (
            self.max_total_tokens is not None
            and usage.total_tokens > self.max_total_tokens
        ):
            violations.append(
                f"total tokens {usage.total_tokens} > {self.max_total_tokens}"
            )
        if self.max_cost is not None and usage.cost > self.max_cost:
            violations.append(f"cost ${usage.cost:.4f} > ${self.max_cost:.4f}")
        return violations

    def check(self, usage: Usage) -> None:
        """Raise :class:`BudgetError` if ``usage`` violates any cap."""
        violations = self.exceeded_by(usage)
        if violations:
            raise BudgetError("Budget exceeded: " + "; ".join(violations))


class BudgetTracker:
    """Accumulates usage and enforces a budget across a run.

    Thread-safe. ``record`` adds a single call's usage to the running total
    and raises :class:`BudgetError` once any cap is crossed, so a node can
    stop work as soon as the budget is spent.
    """

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._tracker = UsageTracker()
        self._lock = threading.Lock()

    @property
    def budget(self) -> Budget:
        """The caps enforced by this tracker."""
        return self._budget

    @property
    def total(self) -> Usage:
        """Aggregate usage recorded so far."""
        return self._tracker.total

    def record(self, usage: Usage) -> Usage:
        """Add ``usage`` to the running total, raising ``BudgetError`` on exceed.

        Returns the updated aggregate usage.
        """
        self._tracker.add(usage)
        total = self._tracker.total
        self._budget.check(total)
        return total

    def check(self) -> None:
        """Raise ``BudgetError`` if the accumulated usage exceeds the budget."""
        self._budget.check(self._tracker.total)

    def reset(self) -> None:
        """Clear recorded usage (e.g. to reuse the tracker per run)."""
        self._tracker.reset()


__all__ = ["Budget", "BudgetError", "BudgetTracker"]
