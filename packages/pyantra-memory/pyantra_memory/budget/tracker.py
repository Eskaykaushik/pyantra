"""Token budget tracking and budget-aware LLM wrapper."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Sequence

from pyantra import LLM, LLMResponse, Message, Usage, UsageTracker


class BudgetExceeded(Exception):
    """Raised when a token budget cap is exceeded."""


@dataclass(frozen=True)
class TokenBudget:
    """Token and cost caps for an LLM session.

    Every field defaults to ``None`` (unlimited). The budget is checked
    after each LLM call; if any cap is crossed, ``BudgetExceeded`` is raised.
    """

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost: float | None = None

    def check(self, usage: Usage, cumulative: Usage) -> None:
        """Raise ``BudgetExceeded`` if usage violates any cap.

        Checks both per-call ``usage`` and ``cumulative`` aggregate.
        """
        violations: list[str] = []
        if (
            self.max_input_tokens is not None
            and cumulative.input_tokens > self.max_input_tokens
        ):
            violations.append(
                f"input tokens {cumulative.input_tokens} > {self.max_input_tokens}"
            )
        if (
            self.max_output_tokens is not None
            and cumulative.output_tokens > self.max_output_tokens
        ):
            violations.append(
                f"output tokens {cumulative.output_tokens} > {self.max_output_tokens}"
            )
        if self.max_cost is not None and cumulative.cost > self.max_cost:
            violations.append(
                f"cost ${cumulative.cost:.4f} > ${self.max_cost:.4f}"
            )
        if violations:
            raise BudgetExceeded("Budget exceeded: " + "; ".join(violations))


class BudgetedLLM:
    """LLM wrapper that enforces a token budget.

    Delegates to the inner LLM, then checks the budget after each call.
    Raises ``BudgetExceeded`` if any cap is crossed.

    Example::

        llm = BudgetedLLM(
            llm=my_llm,
            budget=TokenBudget(max_cost=1.00),
        )
        response = await llm.agenerate(messages)
    """

    def __init__(self, llm: LLM, budget: TokenBudget) -> None:
        self._llm = llm
        self._budget = budget
        self._tracker = UsageTracker()
        self._lock = threading.Lock()

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    @property
    def total_usage(self) -> Usage:
        return self._tracker.total

    def _check_budget(self, usage: Usage) -> None:
        with self._lock:
            self._tracker.add(usage)
            total = self._tracker.total
        self._budget.check(usage, total)

    def generate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        response = self._llm.generate(messages, **kwargs)
        self._check_budget(response.usage)
        return response

    async def agenerate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        response = await self._llm.agenerate(messages, **kwargs)
        self._check_budget(response.usage)
        return response


__all__ = ["BudgetExceeded", "BudgetedLLM", "TokenBudget"]
