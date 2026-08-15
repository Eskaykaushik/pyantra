"""Trajectory and outcome expectations for Pyantra runs.

An :class:`Evaluator` judges a :class:`~pyantra.Run` and produces an
:class:`EvalResult`. The factory functions (``expect_status``,
``expect_ordered``, ...) build evaluators that can be run together with
:func:`evaluate` and aggregated into an :class:`EvalReport`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from pyantra import Run, RunStatus, Usage
from pyantra_eval.judge import LLMJudge

EvalRun: TypeAlias = Run[Any]

@dataclass(frozen=True)
class EvalResult:
    """The verdict of a single evaluator against a run.

    ``score`` and ``max_score`` are ``None`` for purely boolean expectations
    unless they opted into scoring; ``score`` is clamped to ``[0, max_score]``
    when both are set. ``passed`` drives pass/fail regardless of ``score``.
    ``usage`` records LLM cost when the verdict was produced by a judge.
    """

    name: str
    passed: bool
    message: str = ""
    score: float | None = None
    max_score: float | None = None
    usage: Usage | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "score": self.score,
            "max_score": self.max_score,
            "usage": _usage_to_dict(self.usage) if self.usage is not None else None,
        }


class Evaluator(Protocol):
    """Any object that can judge a :class:`~pyantra.Run`."""

    name: str

    def evaluate(self, run: EvalRun) -> EvalResult:
        """Return the verdict for ``run``."""
        ...


class Metric(Evaluator, Protocol):
    """An :class:`Evaluator` that scores a run against a threshold.

    ``threshold`` is ``None`` when the metric is used as a plain assertion;
    ``aggregation`` describes how scores are combined across a suite.
    """

    threshold: float | None
    aggregation: str

    def evaluate(self, run: EvalRun) -> EvalResult:
        """Return the verdict and score for ``run``."""
        ...


def _binary_result(name: str, passed: bool, message: str = "") -> EvalResult:
    """Build a scored result for a boolean expectation (0/1 on a 1 scale)."""
    return EvalResult(
        name=name,
        passed=passed,
        message=message,
        score=1.0 if passed else 0.0,
        max_score=1.0,
    )


def _usage_to_dict(usage: Usage) -> dict[str, object]:
    """Serialize :class:`~pyantra.Usage` for reports."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_tokens": usage.cache_tokens,
        "cost": usage.cost,
        "model": usage.model,
    }


@dataclass(frozen=True)
class EvalReport:
    """Aggregate verdicts from one or more evaluators."""

    results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every result passed."""
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> list[EvalResult]:
        """The results that did not pass."""
        return [result for result in self.results if not result.passed]

    @property
    def scores(self) -> list[float]:
        """The scores of results that recorded one."""
        return [result.score for result in self.results if result.score is not None]

    @property
    def avg_score(self) -> float | None:
        """Mean score across scored results, or ``None`` if there are none."""
        scores = self.scores
        return sum(scores) / len(scores) if scores else None

    def pass_rate(self) -> float:
        """Fraction of results that passed; 1.0 when there are no results."""
        if not self.results:
            return 1.0
        return sum(result.passed for result in self.results) / len(self.results)

    @property
    def total_usage(self) -> Usage:
        """Aggregate LLM usage across scored results (cached hits cost 0)."""
        total = Usage()
        for result in self.results:
            if result.usage is not None:
                total = total + result.usage
        return total

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "pass_rate": self.pass_rate(),
            "avg_score": self.avg_score,
            "total_usage": _usage_to_dict(self.total_usage),
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class StatusExpectation:
    """Pass when the run finished with ``expected`` status."""

    expected: RunStatus
    name: str = "status"

    def evaluate(self, run: EvalRun) -> EvalResult:
        passed = run.status == self.expected
        message = (
            ""
            if passed
            else f"expected status {self.expected.value!r}, got {run.status.value!r}"
        )
        return _binary_result(self.name, passed, message)


@dataclass(frozen=True)
class ErrorExpectation:
    """Pass when the run failed and ``error`` matches expectations.

    Provide ``contains`` (a substring) and/or ``pattern`` (a regular
    expression) to narrow the check; with neither, any failure passes.
    """

    contains: str | None = None
    pattern: str | None = None
    name: str = "error"

    def evaluate(self, run: EvalRun) -> EvalResult:
        if run.status is not RunStatus.FAILED:
            return _binary_result(
                self.name, False, f"expected failure, got {run.status.value!r}"
            )
        error = run.error or ""
        if self.contains is not None and self.contains not in error:
            return _binary_result(
                self.name,
                False,
                f"error {error!r} does not contain {self.contains!r}",
            )
        if self.pattern is not None and not re.search(self.pattern, error):
            return _binary_result(
                self.name,
                False,
                f"error {error!r} does not match pattern {self.pattern!r}",
            )
        return _binary_result(self.name, True)


@dataclass(frozen=True)
class VisitExpectation:
    """Pass when ``node`` appears in the trace at least ``at_least`` times."""

    node: str
    at_least: int = 1
    name: str = "visited"

    def evaluate(self, run: EvalRun) -> EvalResult:
        count = sum(1 for event in run.node_events if event.node == self.node)
        passed = count >= self.at_least
        message = (
            ""
            if passed
            else f"expected node {self.node!r} at least {self.at_least} time(s), "
            f"saw {count}"
        )
        return _binary_result(self.name, passed, message)


@dataclass(frozen=True)
class AbsentExpectation:
    """Pass when ``node`` never appears in the trace."""

    node: str
    name: str = "not_visited"

    def evaluate(self, run: EvalRun) -> EvalResult:
        count = sum(1 for event in run.node_events if event.node == self.node)
        passed = count == 0
        message = "" if passed else f"node {self.node!r} was visited {count} time(s)"
        return _binary_result(self.name, passed, message)


@dataclass(frozen=True)
class OrderExpectation:
    """Pass when ``nodes`` appear in the trace in the given order."""

    nodes: tuple[str, ...]
    name: str = "ordered"

    def evaluate(self, run: EvalRun) -> EvalResult:
        visited = [event.node for event in run.node_events if event.node]
        positions = [visited.index(node) for node in self.nodes if node in visited]
        present = len(positions)
        ordered = present == len(self.nodes) and positions == sorted(positions)
        passed = ordered
        missing = [node for node in self.nodes if node not in visited]
        message = ""
        if not ordered:
            detail = f", missing {missing!r}" if missing else ""
            message = (
                f"expected nodes {list(self.nodes)!r} in order, got {visited!r}{detail}"
            )
        return _binary_result(self.name, passed, message)


@dataclass(frozen=True)
class StepExpectation:
    """Pass when the number of node events is within ``limit``."""

    limit: int
    name: str = "steps"

    def evaluate(self, run: EvalRun) -> EvalResult:
        count = len(run.node_events)
        passed = count <= self.limit
        message = (
            ""
            if passed
            else f"expected at most {self.limit} node event(s), saw {count}"
        )
        return _binary_result(self.name, passed, message)


@dataclass(frozen=True)
class InterruptExpectation:
    """Pass when the run paused for input (status ``PAUSED``)."""

    name: str = "interrupt"

    def evaluate(self, run: EvalRun) -> EvalResult:
        passed = run.status is RunStatus.PAUSED
        message = (
            ""
            if passed
            else f"expected interrupt, got status {run.status.value!r}"
        )
        return _binary_result(self.name, passed, message)


@dataclass(frozen=True)
class CallableExpectation:
    """Pass when ``check`` returns True; ``message`` explains failures.

    ``message`` may be a static string or a callable that receives the run
    and returns a description.
    """

    check: Callable[[EvalRun], bool]
    message: str | Callable[[EvalRun], str]
    name: str

    def evaluate(self, run: EvalRun) -> EvalResult:
        passed = self.check(run)
        detail = (
            self.message(run) if callable(self.message) else str(self.message)
        )
        return _binary_result(self.name, passed, "" if passed else detail)


@dataclass(frozen=True)
class JudgedExpectation:
    """Pass when an LLM judge scores the run's output at or above a threshold.

    ``extract`` pulls the text to judge from the run (defaulting to its final
    state); ``threshold`` defaults to half of the judge's ``max_score``.
    """

    judge: LLMJudge
    extract: Callable[[EvalRun], str] | None = None
    threshold: float | None = None
    name: str = "judged"

    def evaluate(self, run: EvalRun) -> EvalResult:
        text = self.extract(run) if self.extract is not None else _state_text(run)
        verdict = self.judge.judge(text, threshold=self.threshold)
        passed = verdict.passed
        message = ""
        if not passed:
            message = (
                verdict.message
                or f"score {verdict.score:.2f}: {verdict.rationale}"
            )
        return EvalResult(
            self.name,
            passed,
            message,
            score=verdict.score,
            max_score=self.judge.max_score,
            usage=verdict.usage,
        )


def _state_text(run: EvalRun) -> str:
    if isinstance(run.state, str):
        return run.state
    if run.state is None:
        return ""
    return str(run.state)


def expect_status(expected: RunStatus) -> StatusExpectation:
    """Expect the run to finish with ``expected`` status."""
    return StatusExpectation(expected)


def expect_completed() -> StatusExpectation:
    """Expect the run to complete successfully."""
    return StatusExpectation(RunStatus.COMPLETED)


def expect_failed(
    *, contains: str | None = None, pattern: str | None = None
) -> ErrorExpectation:
    """Expect the run to fail, optionally matching the error message."""
    return ErrorExpectation(contains=contains, pattern=pattern)


def expect_error(
    *, contains: str | None = None, pattern: str | None = None
) -> ErrorExpectation:
    """Alias for :func:`expect_failed`."""
    return ErrorExpectation(contains=contains, pattern=pattern)


def expect_visited(node: str, *, at_least: int = 1) -> VisitExpectation:
    """Expect ``node`` to run at least ``at_least`` times."""
    return VisitExpectation(node, at_least=at_least)


def expect_that(
    check: Callable[[EvalRun], bool],
    *,
    message: str | Callable[[EvalRun], str],
    name: str,
) -> CallableExpectation:
    """Expect an arbitrary predicate on the run to hold."""
    return CallableExpectation(check=check, message=message, name=name)


def expect_not_visited(node: str) -> AbsentExpectation:
    """Expect ``node`` never to run."""
    return AbsentExpectation(node)


def expect_ordered(*nodes: str) -> OrderExpectation:
    """Expect ``nodes`` to run in the given order (not necessarily adjacent)."""
    return OrderExpectation(tuple(nodes))


def expect_max_steps(limit: int) -> StepExpectation:
    """Expect at most ``limit`` node events in the trace."""
    return StepExpectation(limit)


def expect_interrupt() -> InterruptExpectation:
    """Expect the run to pause for human input."""
    return InterruptExpectation()


def expect_judged(
    judge: LLMJudge,
    *,
    extract: Callable[[EvalRun], str] | None = None,
    threshold: float | None = None,
    name: str = "judged",
) -> JudgedExpectation:
    """Expect an LLM judge to score the run's output at or above ``threshold``."""
    return JudgedExpectation(
        judge=judge, extract=extract, threshold=threshold, name=name
    )


def evaluate(run: EvalRun, *evaluators: Evaluator) -> EvalReport:
    """Judge ``run`` against ``evaluators`` and return the aggregate report."""
    return EvalReport([evaluator.evaluate(run) for evaluator in evaluators])


__all__ = [
    "AbsentExpectation",
    "CallableExpectation",
    "ErrorExpectation",
    "EvalReport",
    "EvalResult",
    "Evaluator",
    "InterruptExpectation",
    "JudgedExpectation",
    "Metric",
    "OrderExpectation",
    "StatusExpectation",
    "StepExpectation",
    "VisitExpectation",
    "evaluate",
    "expect_completed",
    "expect_error",
    "expect_failed",
    "expect_interrupt",
    "expect_judged",
    "expect_max_steps",
    "expect_not_visited",
    "expect_ordered",
    "expect_status",
    "expect_that",
    "expect_visited",
]
