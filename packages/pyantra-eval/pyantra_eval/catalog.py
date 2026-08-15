"""Scored metric catalog for Pyantra runs.

Deterministic metrics (:class:`TaskCompletionMetric`,
:class:`ToolSelectionMetric`) and G-Eval-style LLM-judged metrics
(:class:`GEvalMetric`, :class:`FaithfulnessMetric`,
:class:`AnswerRelevancyMetric`, :class:`HallucinationMetric`). All implement
the :class:`~pyantra_eval.metrics.Metric` protocol and compose with
:func:`~pyantra_eval.metrics.evaluate` and :class:`~pyantra_eval.suite.SuiteRunner`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from pyantra import LLM, RunStatus, Usage
from pyantra_eval.judge import DEFAULT_PROMPT, LLMJudge, VerdictCache
from pyantra_eval.metrics import EvalResult, EvalRun, _state_text

JudgeInput: TypeAlias = Callable[[EvalRun], str]
ContextInput: TypeAlias = Callable[[EvalRun], object]

_RUBRIC_FAITHFULNESS = (
    "Evaluate whether every factual claim in the response is directly "
    "supported by the provided context. Penalize claims that are not "
    "supported or that contradict the context."
)
_RUBRIC_ANSWER_RELEVANCY = (
    "Evaluate how relevant and on-topic the response is to the question. "
    "Penalize off-topic, vague, or redundant content."
)
_RUBRIC_HALLUCINATION = (
    "Evaluate whether the response contains claims that contradict the "
    "provided context. A response that stays within the context scores "
    "highly; invented or contradictory claims score low."
)

_GEVAL_PROMPT = """Evaluate the following response.

Rubric: {rubric}

Score from 0 to {max_score}, where {max_score} is the best.
Reply with exactly two lines:

Score: <number>
Rationale: <one sentence>

Response to evaluate:
{input}"""

_CONTEXT_KEYS = ("context", "contexts", "retrieval_context")


def _state_context(run: EvalRun) -> object:
    """Pull retrieval context out of a run's state, if it has any."""
    state = run.state
    if isinstance(state, dict):
        for key in _CONTEXT_KEYS:
            if key in state:
                return state[key]
    return ""


def _normalize_context(context: object) -> str:
    """Flatten a context value (string, list, or None) into text."""
    if isinstance(context, str):
        return context
    if isinstance(context, (list, tuple)):
        return "\n".join(str(item) for item in context)
    return "" if context is None else str(context)


@dataclass(frozen=True)
class _JudgeMetric:
    """Shared machinery for G-Eval-style LLM-judged metrics.

    Subclasses set ``name`` and ``rubric`` defaults and may override
    ``_template`` to shape how output and context are presented to the model.
    ``extract`` pulls the output text from the run (default: final state);
    ``extract_context`` pulls the supporting material (default: the state's
    ``context``/``contexts``/``retrieval_context`` keys).
    """

    llm: LLM
    rubric: str
    name: str
    max_score: float = 1.0
    threshold: float | None = None
    aggregation: str = "mean"
    prompt: str = DEFAULT_PROMPT
    cache: VerdictCache | None = None
    extract: JudgeInput | None = None
    extract_context: ContextInput | None = None

    def _judge(self) -> LLMJudge:
        return LLMJudge(
            self.llm,
            rubric=self.rubric,
            max_score=self.max_score,
            prompt=self.prompt,
            cache=self.cache,
        )

    def _input(self, run: EvalRun) -> str:
        output = self.extract(run) if self.extract is not None else _state_text(run)
        context = (
            self.extract_context(run)
            if self.extract_context is not None
            else _state_context(run)
        )
        return self._template(output, _normalize_context(context))

    def _template(self, output: str, context: str) -> str:
        return f"Context:\n{context}\n\nResponse:\n{output}"

    def evaluate(self, run: EvalRun) -> EvalResult:
        effective = self.threshold if self.threshold is not None else self.max_score / 2
        verdict = self._judge().judge(self._input(run), threshold=effective)
        message = ""
        if not verdict.passed:
            message = (
                verdict.message
                or f"score {verdict.score:.2f}: {verdict.rationale}"
            )
        return EvalResult(
            self.name,
            verdict.passed,
            message,
            score=verdict.score,
            max_score=self.max_score,
            usage=verdict.usage,
        )


@dataclass(frozen=True)
class FaithfulnessMetric(_JudgeMetric):
    """Score whether the response's claims are supported by the context."""

    name: str = "faithfulness"
    rubric: str = _RUBRIC_FAITHFULNESS


@dataclass(frozen=True)
class AnswerRelevancyMetric(_JudgeMetric):
    """Score how relevant the response is to the question (in ``context``)."""

    name: str = "answer_relevancy"
    rubric: str = _RUBRIC_ANSWER_RELEVANCY

    def _template(self, output: str, context: str) -> str:
        return f"Question:\n{context}\n\nResponse:\n{output}"


@dataclass(frozen=True)
class HallucinationMetric(_JudgeMetric):
    """Score whether the response contradicts the provided context."""

    name: str = "hallucination"
    rubric: str = _RUBRIC_HALLUCINATION


@dataclass(frozen=True)
class GEvalMetric:
    """G-Eval: score output against a rubric, optionally with explicit steps.

    ``n_samples`` > 1 averages several independent judgments (self-consistency)
    and ``aggregation`` describes how per-case scores combine across a suite.
    ``evaluation_steps`` are rendered into the rubric prompt as numbered steps.
    """

    llm: LLM
    rubric: str
    evaluation_steps: tuple[str, ...] = ()
    max_score: float = 1.0
    threshold: float | None = None
    n_samples: int = 1
    aggregation: str = "mean"
    prompt: str = _GEVAL_PROMPT
    cache: VerdictCache | None = None
    name: str = "g_eval"
    extract: JudgeInput | None = None

    def _judge(self) -> LLMJudge:
        rubric = self.rubric
        if self.evaluation_steps:
            steps = "\n".join(
                f"{index + 1}. {step}"
                for index, step in enumerate(self.evaluation_steps)
            )
            rubric = f"{self.rubric}\n\nEvaluation steps:\n{steps}"
        return LLMJudge(
            self.llm,
            rubric=rubric,
            max_score=self.max_score,
            prompt=self.prompt,
            cache=self.cache,
        )

    def evaluate(self, run: EvalRun) -> EvalResult:
        effective = self.threshold if self.threshold is not None else self.max_score / 2
        text = self.extract(run) if self.extract is not None else _state_text(run)
        judge = self._judge()
        samples = [
            judge.judge(text, threshold=effective)
            for _ in range(max(1, self.n_samples))
        ]
        score = sum(sample.score for sample in samples) / len(samples)
        passed = score >= effective
        usage = Usage()
        for sample in samples:
            usage = usage + sample.usage
        message = ""
        if not passed:
            message = f"score {score:.2f}: {samples[0].rationale}"
        return EvalResult(
            self.name,
            passed,
            message,
            score=score,
            max_score=self.max_score,
            usage=usage,
        )


@dataclass(frozen=True)
class TaskCompletionMetric:
    """Pass when the run completed and produced non-empty output."""

    name: str = "task_completion"
    threshold: float = 1.0
    aggregation: str = "mean"
    extract: JudgeInput | None = None

    def evaluate(self, run: EvalRun) -> EvalResult:
        output = self.extract(run) if self.extract is not None else _state_text(run)
        passed = run.status is RunStatus.COMPLETED and bool(output)
        message = "" if passed else f"run {run.status.value!r} produced no output"
        return EvalResult(
            self.name,
            passed,
            message,
            score=1.0 if passed else 0.0,
            max_score=1.0,
        )


@dataclass(frozen=True)
class ToolSelectionMetric:
    """Score how well the executed node trace matches expectations.

    ``expected_nodes`` must all be visited and ``forbidden_nodes`` must not;
    the score is the fraction of those constraints that held.
    """

    expected_nodes: tuple[str, ...] = ()
    forbidden_nodes: tuple[str, ...] = ()
    name: str = "tool_selection"
    threshold: float = 1.0
    aggregation: str = "mean"

    def evaluate(self, run: EvalRun) -> EvalResult:
        visited = {event.node for event in run.node_events if event.node}
        missing = [node for node in self.expected_nodes if node not in visited]
        intruders = [node for node in self.forbidden_nodes if node in visited]
        total = len(self.expected_nodes) + len(self.forbidden_nodes)
        score = 1.0 - (len(missing) + len(intruders)) / total if total else 1.0
        passed = score >= self.threshold
        message = ""
        if not passed:
            parts = []
            if missing:
                parts.append(f"missing {missing!r}")
            if intruders:
                parts.append(f"unexpected {intruders!r}")
            message = ", ".join(parts)
        return EvalResult(self.name, passed, message, score=score, max_score=1.0)


__all__ = [
    "AnswerRelevancyMetric",
    "ContextInput",
    "FaithfulnessMetric",
    "GEvalMetric",
    "HallucinationMetric",
    "JudgeInput",
    "TaskCompletionMetric",
    "ToolSelectionMetric",
]
