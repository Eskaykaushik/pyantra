"""LLM abstraction: provider contract, usage tracking, and mocks.

See ``packages/pyantra-core/docs/llm.md`` for the full design and what is
deferred to later phases.
"""

from pyantra.llm.base import LLM
from pyantra.llm.mock import MockLLM
from pyantra.llm.types import LLMResponse, Message, Usage
from pyantra.llm.usage import UsageTracker

__all__ = [
    "LLM",
    "LLMResponse",
    "Message",
    "MockLLM",
    "Usage",
    "UsageTracker",
]
