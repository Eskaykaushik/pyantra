"""pytest integration for pyantra-eval.

Declare the ``pyantra_evals`` fixture, register expectations against runs
made inside a test, and the test fails at teardown with a summary if any
expectation did not pass. At session end, all results registered through the
fixture are aggregated into a single summary table (and optionally written to
a JSON report via ``--pyantra-report=PATH``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyantra_eval.metrics import EvalReport, EvalResult, EvalRun, Evaluator, evaluate
from pyantra_eval.report import format_stats_table, results_to_dict

_SESSION_RESULTS: list[EvalResult] = []


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
    _SESSION_RESULTS.extend(report.results)
    if not report.passed:
        pytest.fail(_format_report(report), pytrace=False)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--pyantra-report`` option."""
    parser.addoption(
        "--pyantra-report",
        action="store",
        default=None,
        metavar="PATH",
        help="write an aggregated pyantra-eval JSON report to PATH",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Reset the session-wide result registry."""
    _SESSION_RESULTS.clear()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print the session summary and write the report when requested."""
    if not _SESSION_RESULTS:
        return
    print("\npyantra-eval session summary")
    print(format_stats_table(_SESSION_RESULTS))
    report_path = session.config.getoption("pyantra_report")
    if report_path is not None:
        path = Path(report_path)
        path.write_text(
            json.dumps(results_to_dict("pytest", _SESSION_RESULTS), indent=2),
            encoding="utf-8",
        )
        print(f"pyantra-eval report written to {path}")


__all__ = ["PyantraEvalCollector"]
