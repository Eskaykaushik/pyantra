"""The LLM provider contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pyantra.llm.types import LLMResponse, Message


class LLM(Protocol):
    """The minimal interface every provider adapter implements.

    Nodes interact only with this protocol, so swapping providers never
    changes workflow code. Adapters are responsible for translating
    :class:`Message` into provider-native payloads and for filling in
    :class:`Usage` (including cost) from provider responses.

    See ``packages/pyantra-core/docs/llm.md`` for the design and what is
    deferred to later phases.
    """

    def generate(self, messages: Sequence[Message], **kwargs: object) -> LLMResponse:
        """Generate a completion synchronously."""
        ...

    async def agenerate(
        self, messages: Sequence[Message], **kwargs: object
    ) -> LLMResponse:
        """Generate a completion asynchronously."""
        ...


__all__ = ["LLM", "Message"]
