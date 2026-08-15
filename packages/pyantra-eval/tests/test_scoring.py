from dataclasses import dataclass

import pytest

from pyantra import MockLLM, Run, RunStatus
from pyantra_eval import (
    EvalReport,
    EvalResult,
    JudgeResult,
    LLMJudge,
    evaluate,
    expect_completed,
    expect_judged,
)


class MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, JudgeResult] = {}

    def get(self, key: str) -> JudgeResult | None:
        return self._store.get(key)

    def set(self, key: str, result: JudgeResult) -> None:
        self._store[key] = result


@dataclass(frozen=True)
class ScoredMetric:
    name: str
    threshold: float = 0.5
    aggregation: str = "mean"

    def evaluate(self, run: Run) -> EvalResult:
        passed = run.status is RunStatus.COMPLETED
        return EvalResult(
            self.name,
            passed=passed,
            score=0.9 if passed else 0.2,
            max_score=1.0,
        )


def _run(status: RunStatus = RunStatus.COMPLETED, state: object = None) -> Run:
    return Run(run_id="r1", status=status, state=state)


def test_binary_expectations_report_scores() -> None:
    result = expect_completed().evaluate(_run())
    assert isinstance(result, EvalResult)
    assert result.score == 1.0
    assert result.max_score == 1.0
    failed = expect_completed().evaluate(_run(RunStatus.FAILED))
    assert failed.score == 0.0
    assert not failed.passed


def test_eval_result_to_dict_includes_scores() -> None:
    result = EvalResult("m", passed=True, score=0.8, max_score=1.0)
    assert result.to_dict() == {
        "name": "m",
        "passed": True,
        "message": "",
        "score": 0.8,
        "max_score": 1.0,
        "usage": None,
    }


def test_eval_result_to_dict_includes_usage() -> None:
    from pyantra import Usage

    result = EvalResult(
        "m", passed=True, score=0.8, max_score=1.0, usage=Usage(input_tokens=5)
    )
    assert result.to_dict()["usage"] == {
        "input_tokens": 5,
        "output_tokens": 0,
        "cache_tokens": 0,
        "cost": 0.0,
        "model": "",
    }


def test_report_scores_avg_and_pass_rate() -> None:
    report = EvalReport(
        [
            EvalResult("a", passed=True, score=1.0, max_score=1.0),
            EvalResult("b", passed=False, score=0.3, max_score=1.0),
            EvalResult("c", passed=True),
        ]
    )
    assert report.scores == [1.0, 0.3]
    assert report.avg_score == pytest.approx(0.65)
    assert report.pass_rate() == pytest.approx(2 / 3)
    assert report.avg_score == report.to_dict()["avg_score"]
    assert report.pass_rate() == report.to_dict()["pass_rate"]


def test_report_avg_score_none_without_scores() -> None:
    report = EvalReport([EvalResult("a", passed=True), EvalResult("b", passed=True)])
    assert report.scores == []
    assert report.avg_score is None
    assert report.pass_rate() == 1.0


def test_report_pass_rate_empty_is_one() -> None:
    assert EvalReport().pass_rate() == 1.0


def test_metric_protocol_composes_with_evaluators() -> None:
    report = evaluate(
        _run(),
        expect_completed(),
        ScoredMetric(name="quality", threshold=0.5),
        ScoredMetric(name="speed", threshold=0.8),
    )
    assert report.passed
    assert report.scores == [1.0, 0.9, 0.9]
    assert report.avg_score == pytest.approx(2.8 / 3)
    assert report.to_dict()["avg_score"] == pytest.approx(2.8 / 3)


def test_metric_failure_keeps_score_in_failures() -> None:
    metric = ScoredMetric(name="quality", threshold=0.5)
    report = evaluate(_run(RunStatus.FAILED), metric)
    assert not report.passed
    assert report.scores == [0.2]
    assert [f.name for f in report.failures] == ["quality"]


def test_expect_judged_records_score_and_max() -> None:
    llm = MockLLM(responses=["Score: 7/10\nRationale: Good."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    report = evaluate(_run(state="answer"), expect_judged(judge, threshold=6.0))
    result = report.results[0]
    assert result.score == 7.0
    assert result.max_score == 10.0
    assert report.avg_score == pytest.approx(7.0)


def test_judge_cache_hit_skips_model_and_zeroes_usage() -> None:
    llm = MockLLM(responses=["Score: 9/10\nRationale: Good."])
    cache = MemoryCache()
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0, cache=cache)
    first = judge.judge("same input")
    second = judge.judge("same input")
    assert first.score == second.score == 9.0
    assert len(llm.recorded_calls) == 1
    assert second.usage.total_tokens == 0
    assert second.usage.cost == 0.0


def test_judge_cache_miss_still_records_usage() -> None:
    llm = MockLLM(responses=["Score: 5/10\nRationale: OK."], cost=0.01)
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0, cache=MemoryCache())
    result = judge.judge("input a")
    assert len(llm.recorded_calls) == 1
    assert result.usage.cost == 0.01


def test_judge_cache_keyed_on_text() -> None:
    llm = MockLLM(
        responses=["Score: 9/10\nRationale: Good.", "Score: 3/10\nRationale: Bad."]
    )
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0, cache=MemoryCache())
    assert judge.judge("one").score == 9.0
    assert judge.judge("two").score == 3.0
    assert len(llm.recorded_calls) == 2


async def test_ajudge_uses_cache() -> None:
    llm = MockLLM(responses=["Score: 8/10\nRationale: Fine."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0, cache=MemoryCache())
    first = await judge.ajudge("async input")
    second = await judge.ajudge("async input")
    assert first.score == second.score == 8.0
    assert len(llm.recorded_calls) == 1
