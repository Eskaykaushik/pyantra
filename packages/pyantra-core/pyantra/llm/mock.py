"""A deterministic, scripted LLM for tests, examples, and future replay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pyantra.llm.types import LLMResponse, Message, Usage


@dataclass
class MockLLM:
    """A scripted LLM that never makes a network call.

    ``responses`` is consumed in order and cycles back to the start once
    exhausted. Each call reports fixed token counts (and optional cost) so
    usage aggregation, budgets, and cost logic can be tested without a real
    provider. The recorded messages are kept on :attr:`recorded_calls` for
    asserting what a node actually sent.
    """

    responses: list[str] = field(default_factory=list)
    input_tokens: int = 10
    output_tokens: int = 10
    cost: float = 0.0
    model: str = "mock"

    def __post_init__(self) -> None:
        self.recorded_calls: list[list[Message]] = []
        self._cursor = 0

    def generate(self, messages: Sequence[Message], **kwargs: object) -> LLMResponse:
        return self._respond(messages)

    async def agenerate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        return self._respond(messages)

    def _respond(self, messages: Sequence[Message]) -> LLMResponse:
        self.recorded_calls.append(list(messages))
        if not self.responses:
            raise ValueError("MockLLM has no scripted responses.")
        content = self.responses[self._cursor % len(self.responses)]
        self._cursor += 1
        return LLMResponse(
            content=content,
            usage=Usage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost=self.cost,
                model=self.model,
            ),
        )


__all__ = ["MockLLM"]
