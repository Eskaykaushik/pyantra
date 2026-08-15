"""LLM-as-judge scoring.

``LLMJudge`` wraps a :class:`~pyantra.LLM` provider and prompts it to score
an arbitrary piece of output against a rubric. It parses a numeric verdict
out of the model's response (accepting ``Score: 3`` or ``Score: 7/10``),
clamps it to the configured range, and records the reasoning and usage.
An optional :class:`VerdictCache` short-circuits repeat judgments; cached
verdicts report empty :class:`~pyantra.Usage` since no call was made.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

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


class VerdictCache(Protocol):
    """A persistent store for judge verdicts keyed by a string.

    Implementations must be safe for concurrent ``get``/``set`` because a
    suite may judge many cases in parallel.
    """

    def get(self, key: str) -> JudgeResult | None:
        """Return the cached verdict for ``key``, or ``None``."""
        ...

    def set(self, key: str, result: JudgeResult) -> None:
        """Store ``result`` under ``key``."""
        ...


class DiskCache:
    """A file-backed :class:`VerdictCache` rooted at ``directory``.

    Each verdict is one JSON file named ``<key>.json``, written atomically
    (temp file + rename) and guarded by a process-local lock, so concurrent
    judges within a run are safe.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._lock = threading.Lock()

    def get(self, key: str) -> JudgeResult | None:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        return _result_from_dict(data)

    def set(self, key: str, result: JudgeResult) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        data = _result_to_dict(result)
        path = self._path(key)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with self._lock:
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(path)

    def _path(self, key: str) -> Path:
        return self._directory / f"{key}.json"


def _result_to_dict(result: JudgeResult) -> dict[str, object]:
    return {
        "score": result.score,
        "rationale": result.rationale,
        "passed": result.passed,
        "message": result.message,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cache_tokens": result.usage.cache_tokens,
            "cost": result.usage.cost,
            "model": result.usage.model,
        },
    }


def _result_from_dict(data: object) -> JudgeResult:
    if not isinstance(data, dict):
        return JudgeResult(score=0.0, passed=False, message="corrupt cache entry")
    usage_data = data.get("usage")
    usage = (
        Usage(
            input_tokens=_as_int(usage_data.get("input_tokens")),
            output_tokens=_as_int(usage_data.get("output_tokens")),
            cache_tokens=_as_int(usage_data.get("cache_tokens")),
            cost=_as_float(usage_data.get("cost")),
            model=str(usage_data.get("model", "")),
        )
        if isinstance(usage_data, dict)
        else Usage()
    )
    return JudgeResult(
        score=_as_float(data.get("score")),
        rationale=str(data.get("rationale", "")),
        passed=bool(data.get("passed")),
        usage=usage,
        message=str(data.get("message", "")),
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


class LLMJudge:
    """Score output with an LLM against a rubric.

    ``prompt`` may be overridden; it is formatted with ``rubric``,
    ``max_score``, and ``input``. Extra keyword arguments to ``judge`` are
    forwarded to the provider's ``generate``/``agenerate``. When ``cache``
    is provided, identical (prompt, rubric, max_score, text, threshold)
    judgments are served from the cache instead of calling the model.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        rubric: str,
        max_score: float = 1.0,
        prompt: str = DEFAULT_PROMPT,
        cache: VerdictCache | None = None,
    ) -> None:
        self.llm = llm
        self.rubric = rubric
        self.max_score = max_score
        self.prompt = prompt
        self.cache = cache

    def _prompt(self, text: str) -> str:
        return self.prompt.format(
            rubric=self.rubric, max_score=self.max_score, input=text
        )

    def _key(self, text: str, threshold: float) -> str:
        payload = json.dumps(
            {
                "prompt": self.prompt,
                "rubric": self.rubric,
                "max_score": self.max_score,
                "text": text,
                "threshold": threshold,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cached(self, text: str, threshold: float) -> JudgeResult | None:
        if self.cache is None:
            return None
        return self.cache.get(self._key(text, threshold))

    def _store(self, text: str, threshold: float, result: JudgeResult) -> None:
        if self.cache is not None:
            self.cache.set(
                self._key(text, threshold), replace(result, usage=Usage())
            )

    def judge(
        self, text: str, *, threshold: float | None = None, **kwargs: object
    ) -> JudgeResult:
        """Score ``text`` synchronously, returning the verdict."""
        effective = threshold if threshold is not None else self.max_score / 2
        cached = self._cached(text, effective)
        if cached is not None:
            return cached
        response = self.llm.generate(
            [Message(role="user", content=self._prompt(text))], **kwargs
        )
        result = _parse(response, self.max_score, effective)
        self._store(text, effective, result)
        return result

    async def ajudge(
        self, text: str, *, threshold: float | None = None, **kwargs: object
    ) -> JudgeResult:
        """Score ``text`` asynchronously, returning the verdict."""
        effective = threshold if threshold is not None else self.max_score / 2
        cached = self._cached(text, effective)
        if cached is not None:
            return cached
        response = await self.llm.agenerate(
            [Message(role="user", content=self._prompt(text))], **kwargs
        )
        result = _parse(response, self.max_score, effective)
        self._store(text, effective, result)
        return result


__all__ = ["DEFAULT_PROMPT", "DiskCache", "JudgeResult", "LLMJudge", "VerdictCache"]
