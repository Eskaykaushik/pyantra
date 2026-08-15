"""Aggregation and formatting helpers for eval results.

Both the :mod:`pyantra_eval.cli` and the pytest plugin use these to turn flat
lists of :class:`~pyantra_eval.metrics.EvalResult` into per-metric statistics
and human-readable summaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, quantiles

from pyantra import Usage
from pyantra_eval.metrics import EvalResult, _usage_to_dict
from pyantra_eval.suite import SuiteReport


@dataclass(frozen=True)
class MetricStats:
    """Aggregate statistics for a single metric across results."""

    name: str
    count: int
    passed: int
    pass_rate: float
    mean: float | None
    p50: float | None
    p95: float | None


def _percentile(sorted_scores: Sequence[float], q: float) -> float:
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    boundaries = quantiles(sorted_scores, n=100, method="inclusive")
    return boundaries[int(q * 100) - 1]


def aggregate_metrics(results: Sequence[EvalResult]) -> list[MetricStats]:
    """Group ``results`` by metric name and summarize each group."""
    by_name: dict[str, list[EvalResult]] = {}
    for result in results:
        by_name.setdefault(result.name, []).append(result)

    stats: list[MetricStats] = []
    for name in sorted(by_name):
        entries = by_name[name]
        scores = sorted(
            entry.score for entry in entries if entry.score is not None
        )
        passed = sum(entry.passed for entry in entries)
        stats.append(
            MetricStats(
                name=name,
                count=len(entries),
                passed=passed,
                pass_rate=passed / len(entries),
                mean=fmean(scores) if scores else None,
                p50=_percentile(scores, 0.5) if scores else None,
                p95=_percentile(scores, 0.95) if scores else None,
            )
        )
    return stats


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def format_stats_table(results: Sequence[EvalResult]) -> str:
    """Render per-metric statistics as an aligned table."""
    header = (
        f"{'metric':<18} {'pass':<5} {'rate':>5} {'mean':>6} "
        f"{'p50':>6} {'p95':>6} {'n':>4}"
    )
    rows = [header]
    for stats in aggregate_metrics(results):
        rows.append(
            f"{stats.name:<18} "
            f"{'OK' if stats.pass_rate == 1.0 else 'FAIL':<5} "
            f"{stats.pass_rate:>5.2f} {_fmt(stats.mean):>6} "
            f"{_fmt(stats.p50):>6} {_fmt(stats.p95):>6} {stats.count:>4}"
        )
    return "\n".join(rows)


def total_usage(results: Sequence[EvalResult]) -> Usage:
    """Aggregate LLM usage across ``results``."""
    total = Usage()
    for result in results:
        if result.usage is not None:
            total = total + result.usage
    return total


def results_to_dict(
    name: str, results: Sequence[EvalResult]
) -> dict[str, object]:
    """Serialize a flat result list into a report dict."""
    scored = [result.score for result in results if result.score is not None]
    return {
        "tool": "pyantra-eval",
        "dataset": name,
        "passed": all(result.passed for result in results),
        "pass_rate": (
            sum(result.passed for result in results) / len(results)
            if results
            else 1.0
        ),
        "avg_score": fmean(scored) if scored else None,
        "total_usage": _usage_to_dict(total_usage(results)),
        "metrics": [
            {
                "name": stats.name,
                "count": stats.count,
                "passed": stats.passed,
                "pass_rate": stats.pass_rate,
                "mean": stats.mean,
                "p50": stats.p50,
                "p95": stats.p95,
            }
            for stats in aggregate_metrics(results)
        ],
        "results": [result.to_dict() for result in results],
    }


def format_suite_report(report: SuiteReport) -> str:
    """Render a suite report as a human-readable summary."""
    results = [
        result
        for case in report.results
        for result in case.report.results
    ]
    lines = [f"Dataset: {report.dataset_name}", format_stats_table(results)]
    usage = total_usage(results)
    lines.append(
        f"Overall: pass_rate={report.pass_rate():.2f} "
        f"avg_score={_fmt(report.avg_score)} "
        f"cost=${usage.cost:.4f} ({usage.total_tokens} tokens)"
    )
    return "\n".join(lines)


__all__ = [
    "MetricStats",
    "aggregate_metrics",
    "format_stats_table",
    "format_suite_report",
    "results_to_dict",
    "total_usage",
]
