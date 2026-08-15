import pytest
from pyantra_guard import Budget, BudgetError, BudgetTracker

from pyantra import Usage


def test_budget_no_caps_allows_any_usage() -> None:
    budget = Budget()
    assert budget.exceeded_by(Usage(input_tokens=10_000, cost=100.0)) == []
    budget.check(Usage(input_tokens=10_000, cost=100.0))


def test_budget_reports_every_violation() -> None:
    budget = Budget(max_input_tokens=100, max_output_tokens=50, max_total_tokens=200)
    violations = budget.exceeded_by(
        Usage(input_tokens=101, output_tokens=51, cache_tokens=200)
    )
    assert len(violations) == 3
    assert any("input tokens" in v for v in violations)
    assert any("output tokens" in v for v in violations)
    assert any("total tokens" in v for v in violations)


def test_budget_check_raises() -> None:
    budget = Budget(max_total_tokens=10)
    with pytest.raises(BudgetError, match="Budget exceeded"):
        budget.check(Usage(input_tokens=11))


def test_budget_cost_cap() -> None:
    budget = Budget(max_cost=1.0)
    with pytest.raises(BudgetError, match=r"cost \$1\.0001 > \$1\.0000"):
        budget.check(Usage(input_tokens=10, cost=1.0001))


def test_tracker_accumulates_and_raises_on_cap() -> None:
    tracker = BudgetTracker(Budget(max_total_tokens=100))
    total = tracker.record(Usage(input_tokens=60))
    assert total.total_tokens == 60
    with pytest.raises(BudgetError, match="total tokens"):
        tracker.record(Usage(input_tokens=60))
    assert tracker.total.total_tokens == 120


def test_tracker_total_after_reset() -> None:
    tracker = BudgetTracker(Budget(max_total_tokens=1000))
    tracker.record(Usage(input_tokens=50))
    tracker.reset()
    assert tracker.total.total_tokens == 0
    tracker.record(Usage(input_tokens=10))
    assert tracker.total.total_tokens == 10
