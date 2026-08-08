"""LLM value types: messages, usage accounting, and responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """A single chat message passed to a model.

    ``role`` is one of ``"system"``, ``"user"``, ``"assistant"``, or
    ``"tool"``. Provider adapters translate this into their native payloads.
    """

    role: str
    content: str


@dataclass(frozen=True)
class Usage:
    """Token and cost accounting for a single model invocation.

    ``cost`` is expressed in USD. When the provider does not report it,
    adapters compute it from their pricing tables. ``model`` names the model
    that produced the usage; it is kept for observability across providers.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    cost: float = 0.0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all categories."""
        return self.input_tokens + self.output_tokens + self.cache_tokens

    def __add__(self, other: Usage) -> Usage:
        """Combine two usage records (for aggregation across a run)."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_tokens=self.cache_tokens + other.cache_tokens,
            cost=self.cost + other.cost,
            model=self.model or other.model,
        )


@dataclass(frozen=True)
class LLMResponse:
    """A model generation together with its usage accounting."""

    content: str
    usage: Usage
