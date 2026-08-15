"""pytest integration for pyantra-eval.

Declare the ``pyantra_evals`` fixture, register expectations against runs
made inside a test, and the test fails at teardown with a summary if any
expectation did not pass.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pyantra_eval.metrics import EvalReport, EvalResult, EvalRun, Evaluator, evaluate


class PyantraEvalCollector:
    """Accumulates eval results declared during a test."""

    def __init__(self) -> None:
        self._results: list[EvalResult] = []

    def expect(self, run: EvalRun, *evaluators: Evaluator) -> EvalReport:
        """Judge ``run`` against ``evaluators`` and remember the results."""
        report = evaluate(run, *evaluators)
        self._results.extend(report.results)
        return report

    @property
    def results(self) -> list[EvalResult]:
        """All results recorded so far."""
        return list(self._results)

    def report(self) -> EvalReport:
        """Aggregate all recorded results."""
        return EvalReport(self._results)


def _format_report(report: EvalReport) -> str:
    lines = ["pyantra-eval expectations failed:"]
    lines.extend(f"  - {result.name}: {result.message}" for result in report.failures)
    return "\n".join(lines)


@pytest.fixture
def pyantra_evals() -> Iterator[PyantraEvalCollector]:
    """Fixture that fails the test if any registered expectation fails."""
    collector = PyantraEvalCollector()
    yield collector
    report = collector.report()
    if not report.passed:
        pytest.fail(_format_report(report), pytrace=False)


__all__ = ["PyantraEvalCollector"]
