"""LLM-as-judge scoring.

``LLMJudge`` wraps a :class:`~pyantra.LLM` provider and prompts it to score
an arbitrary piece of output against a rubric. It parses a numeric verdict
out of the model's response (accepting ``Score: 3`` or ``Score: 7/10``),
clamps it to the configured range, and records the reasoning and usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pyantra import LLM, LLMResponse, Message, Usage

DEFAULT_PROMPT = """You are a strict evaluator.

Rubric: {rubric}

Score the response from 0 to {max_score}, where {max_score} is the best.
Reply with exactly two lines:

Score: <number>
Rationale: <one sentence>

Response to evaluate:
{input}"""


@dataclass(frozen=True)
class JudgeResult:
    """The verdict of an LLM judge on a piece of output.

    ``score`` is clamped to ``[0, max_score]``. ``passed`` compares against
    the threshold used when judging. ``message`` is set when the score could
    not be parsed from the model's response.
    """

    score: float
    rationale: str = ""
    passed: bool = True
    usage: Usage = field(default_factory=Usage)
    message: str = ""


_SCORE_RE = re.compile(
    r"score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*/\s*([0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE,
)
_RATIONALE_RE = re.compile(r"rationale\s*[:=]\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse(response: LLMResponse, max_score: float, threshold: float) -> JudgeResult:
    """Extract a :class:`JudgeResult` from a model response."""
    content = response.content
    match = _SCORE_RE.search(content)
    if match is None:
        return JudgeResult(
            score=0.0,
            passed=False,
            usage=response.usage,
            message=f"could not parse a score from {content!r}",
        )
    numerator = float(match.group(1))
    denominator = float(match.group(2)) if match.group(2) is not None else max_score
    raw = numerator / denominator
    score = max(0.0, min(raw * max_score, max_score))
    rationale_match = _RATIONALE_RE.search(content)
    rationale = rationale_match.group(1).strip() if rationale_match else content.strip()
    return JudgeResult(
        score=score,
        rationale=rationale,
        passed=score >= threshold,
        usage=response.usage,
    )


class LLMJudge:
    """Score output with an LLM against a rubric.

    ``prompt`` may be overridden; it is formatted with ``rubric``,
    ``max_score``, and ``input``. Extra keyword arguments to ``judge`` are
    forwarded to the provider's ``generate``/``agenerate``.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        rubric: str,
        max_score: float = 1.0,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        self.llm = llm
        self.rubric = rubric
        self.max_score = max_score
        self.prompt = prompt

    def _prompt(self, text: str) -> str:
        return self.prompt.format(
            rubric=self.rubric, max_score=self.max_score, input=text
        )

    def judge(
        self, text: str, *, threshold: float | None = None, **kwargs: object
    ) -> JudgeResult:
        """Score ``text`` synchronously, returning the verdict."""
        effective = threshold if threshold is not None else self.max_score / 2
        response = self.llm.generate(
            [Message(role="user", content=self._prompt(text))], **kwargs
        )
        return _parse(response, self.max_score, effective)

    async def ajudge(
        self, text: str, *, threshold: float | None = None, **kwargs: object
    ) -> JudgeResult:
        """Score ``text`` asynchronously, returning the verdict."""
        effective = threshold if threshold is not None else self.max_score / 2
        response = await self.llm.agenerate(
            [Message(role="user", content=self._prompt(text))], **kwargs
        )
        return _parse(response, self.max_score, effective)


__all__ = ["DEFAULT_PROMPT", "JudgeResult", "LLMJudge"]
