"""Tests for token budget tracking."""

from __future__ import annotations

from typing import Sequence

from pyantra import LLM, LLMResponse, Message, Usage
from pyantra_memory.budget.tracker import BudgetExceeded, BudgetedLLM, TokenBudget


class _MockLLM(LLM):
    def __init__(self, usage: Usage | None = None) -> None:
        self._usage = usage or Usage(input_tokens=10, output_tokens=5, model="mock")

    def generate(self, messages: Sequence[Message], **kwargs: object) -> LLMResponse:
        return LLMResponse(content="ok", usage=self._usage)

    async def agenerate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        return LLMResponse(content="ok", usage=self._usage)


class TestTokenBudget:
    def test_no_limits(self) -> None:
        budget = TokenBudget()
        usage = Usage(input_tokens=1000, output_tokens=500, cost=99.99)
        budget.check(usage, usage)

    def test_input_tokens_ok(self) -> None:
        budget = TokenBudget(max_input_tokens=100)
        usage = Usage(input_tokens=50)
        budget.check(usage, usage)

    def test_input_tokens_exceeded(self) -> None:
        budget = TokenBudget(max_input_tokens=10)
        usage = Usage(input_tokens=50)
        try:
            budget.check(usage, usage)
            assert False, "Should have raised BudgetExceeded"
        except BudgetExceeded as e:
            assert "input tokens" in str(e)

    def test_output_tokens_exceeded(self) -> None:
        budget = TokenBudget(max_output_tokens=5)
        usage = Usage(output_tokens=20)
        try:
            budget.check(usage, usage)
            assert False, "Should have raised BudgetExceeded"
        except BudgetExceeded as e:
            assert "output tokens" in str(e)

    def test_cost_exceeded(self) -> None:
        budget = TokenBudget(max_cost=0.50)
        usage = Usage(cost=1.00)
        try:
            budget.check(usage, usage)
            assert False, "Should have raised BudgetExceeded"
        except BudgetExceeded as e:
            assert "cost" in str(e)

    def test_cumulative_check(self) -> None:
        budget = TokenBudget(max_input_tokens=15)
        single = Usage(input_tokens=10)
        cumulative = Usage(input_tokens=20)
        try:
            budget.check(single, cumulative)
            assert False, "Should have raised BudgetExceeded"
        except BudgetExceeded:
            pass


class TestBudgetedLLM:
    def test_within_budget(self) -> None:
        llm = BudgetedLLM(
            llm=_MockLLM(Usage(input_tokens=10, output_tokens=5, cost=0.01)),
            budget=TokenBudget(max_cost=1.00),
        )
        response = llm.generate([Message(role="user", content="hi")])
        assert response.content == "ok"
        assert llm.total_usage.input_tokens == 10

    def test_exceeds_budget(self) -> None:
        llm = BudgetedLLM(
            llm=_MockLLM(Usage(input_tokens=100, cost=5.00)),
            budget=TokenBudget(max_cost=1.00),
        )
        try:
            llm.generate([Message(role="user", content="hi")])
            assert False, "Should have raised BudgetExceeded"
        except BudgetExceeded:
            pass

    async def test_async_within_budget(self) -> None:
        llm = BudgetedLLM(
            llm=_MockLLM(Usage(input_tokens=10, cost=0.01)),
            budget=TokenBudget(max_cost=1.00),
        )
        response = await llm.agenerate([Message(role="user", content="hi")])
        assert response.content == "ok"

    async def test_async_exceeds_budget(self) -> None:
        llm = BudgetedLLM(
            llm=_MockLLM(Usage(input_tokens=100, cost=5.00)),
            budget=TokenBudget(max_cost=1.00),
        )
        try:
            await llm.agenerate([Message(role="user", content="hi")])
            assert False, "Should have raised BudgetExceeded"
        except BudgetExceeded:
            pass

    def test_cumulative_tracking(self) -> None:
        llm = BudgetedLLM(
            llm=_MockLLM(Usage(input_tokens=10, cost=0.01)),
            budget=TokenBudget(max_cost=1.00),
        )
        llm.generate([Message(role="user", content="hi")])
        llm.generate([Message(role="user", content="hi")])
        assert llm.total_usage.input_tokens == 20
        assert llm.total_usage.cost == 0.02

    def test_cumulative_exceeds(self) -> None:
        llm = BudgetedLLM(
            llm=_MockLLM(Usage(input_tokens=10, cost=0.60)),
            budget=TokenBudget(max_cost=1.00),
        )
        llm.generate([Message(role="user", content="hi")])
        try:
            llm.generate([Message(role="user", content="hi")])
            assert False, "Should have raised BudgetExceeded"
        except BudgetExceeded:
            pass
