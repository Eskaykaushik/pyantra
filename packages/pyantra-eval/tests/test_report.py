import pytest

from pyantra import Usage
from pyantra_eval.metrics import EvalResult
from pyantra_eval.report import (
    aggregate_metrics,
    format_stats_table,
    results_to_dict,
    total_usage,
)


def _result(
    name: str, passed: bool, score: float | None = None, cost: float = 0.0
) -> EvalResult:
    usage = Usage(cost=cost) if cost else None
    return EvalResult(name, passed, score=score, max_score=1.0, usage=usage)


def test_aggregate_metrics_groups_and_scores() -> None:
    results = [
        _result("faithfulness", True, 0.9),
        _result("faithfulness", False, 0.4),
        _result("task_completion", True, 1.0),
    ]
    stats = aggregate_metrics(results)
    assert [s.name for s in stats] == ["faithfulness", "task_completion"]
    faith = stats[0]
    assert faith.count == 2
    assert faith.passed == 1
    assert faith.pass_rate == pytest.approx(0.5)
    assert faith.mean == pytest.approx(0.65)
    assert faith.p50 == pytest.approx(0.65)
    assert faith.p95 == pytest.approx(0.875)


def test_aggregate_metrics_without_scores() -> None:
    stats = aggregate_metrics([_result("x", True)])
    assert stats[0].mean is None
    assert stats[0].p50 is None
    assert stats[0].p95 is None
    assert stats[0].pass_rate == 1.0


def test_format_stats_table_renders_rows() -> None:
    table = format_stats_table([_result("m", True, 0.8), _result("m", True, 1.0)])
    assert "metric" in table
    assert "m" in table
    assert "OK" in table


def test_total_usage_sums_costs() -> None:
    usage = total_usage([_result("a", True, cost=0.01), _result("b", False, cost=0.02)])
    assert usage.cost == pytest.approx(0.03)


def test_results_to_dict_shape() -> None:
    results = [_result("a", True, cost=0.01), _result("c", True, score=0.9)]
    data = results_to_dict("smoke", results)
    assert data["dataset"] == "smoke"
    assert data["passed"] is True
    assert data["pass_rate"] == 1.0
    assert data["avg_score"] == pytest.approx(0.9)
    assert data["total_usage"]["cost"] == pytest.approx(0.01)
    assert data["metrics"][0]["name"] == "a"
    assert len(data["results"]) == 2
