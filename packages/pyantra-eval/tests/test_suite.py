import asyncio
import json

from pyantra import MockLLM, Run, RunStatus
from pyantra_eval import (
    EvalCase,
    EvalDataset,
    LLMJudge,
    SuiteRunner,
    expect_completed,
    expect_judged,
)


def _app(input_value: object) -> Run:
    ok = isinstance(input_value, dict) and input_value.get("ok") is True
    return Run(
        run_id="r1",
        status=RunStatus.COMPLETED if ok else RunStatus.FAILED,
        state=input_value,
    )


def _dataset() -> EvalDataset:
    return (
        EvalDataset("quality")
        .add(EvalCase("good", input={"ok": True}))
        .add(EvalCase("bad", input={"ok": False}))
    )


def test_case_and_dataset_basics() -> None:
    case = EvalCase("c1", input={"q": 1}, expected="yes", metadata={"cat": "x"})
    dataset = EvalDataset("d", [case])
    assert len(dataset) == 1
    assert dataset[0] is case
    assert [c.id for c in dataset] == ["c1"]
    assert case.to_dict()["expected"] == "yes"
    assert case.to_dict()["metadata"] == {"cat": "x"}


def test_dataset_from_dict() -> None:
    dataset = EvalDataset.from_dict(
        "d",
        [
            {"id": "a", "input": 1, "expected": 2, "metadata": {"k": "v"}},
            {"input": "x"},
        ],
    )
    assert len(dataset) == 2
    assert dataset[0].id == "a"
    assert dataset[0].expected == 2
    assert dataset[0].metadata == {"k": "v"}
    assert dataset[1].id == "case-1"
    assert dataset[1].input == "x"
    assert dataset.to_dict()["name"] == "d"


def test_suite_run_passes_and_fails_correctly() -> None:
    report = SuiteRunner(_app, _dataset(), [expect_completed()]).run()
    assert not report.passed
    assert report.pass_rate() == 0.5
    assert [f.case_id for f in report.failures] == ["bad"]


def test_suite_run_all_pass() -> None:
    dataset = EvalDataset("d").add(EvalCase("c1", input={"ok": True}))
    report = SuiteRunner(_app, dataset, [expect_completed()]).run()
    assert report.passed
    assert report.pass_rate() == 1.0
    assert report.failures == []


def test_suite_report_avg_score_and_serialization() -> None:
    llm = MockLLM(responses=["Score: 8/10\nRationale: Good."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    dataset = EvalDataset("d").add(
        EvalCase("c1", input={"ok": True}, expected="target")
    )
    runner = SuiteRunner(
        _app,
        dataset,
        [
            expect_completed(),
            expect_judged(
                judge, extract=lambda run: str(run.state), threshold=5.0
            ),
        ],
    )
    report = runner.run()
    assert report.passed
    assert report.avg_score is not None
    assert report.avg_score == (1.0 + 8.0) / 2
    assert report.total_usage.input_tokens == llm.input_tokens
    payload = json.dumps(report.to_dict())
    assert '"passed": true' in payload


async def test_arun_with_async_app() -> None:
    async def async_app(input_value: object) -> Run:
        await asyncio.sleep(0.001)
        return Run(
            run_id=str(input_value), status=RunStatus.COMPLETED, state=input_value
        )

    dataset = EvalDataset("d").add(EvalCase("c1", input="v1")).add(
        EvalCase("c2", input="v2")
    )
    report = await SuiteRunner(async_app, dataset, [expect_completed()]).arun()
    assert report.passed
    assert [r.case_id for r in report.results] == ["c1", "c2"]


def test_run_with_async_app() -> None:
    async def async_app(input_value: object) -> Run:
        return Run(
            run_id=str(input_value), status=RunStatus.COMPLETED, state=input_value
        )

    dataset = EvalDataset("d").add(EvalCase("c1", input="v1"))
    report = SuiteRunner(async_app, dataset, [expect_completed()]).run()
    assert report.passed


def test_run_with_concurrency_threads() -> None:
    dataset = EvalDataset("d")
    for i in range(6):
        dataset.add(EvalCase(f"c{i}", input={"ok": i % 2 == 0}))
    report = SuiteRunner(_app, dataset, [expect_completed()], concurrency=3).run()
    assert report.pass_rate() == 0.5
    assert len(report.results) == 6


async def test_arun_honors_concurrency() -> None:
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def app(input_value: object) -> Run:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return Run(run_id=str(input_value), status=RunStatus.COMPLETED)

    dataset = EvalDataset("d")
    for i in range(8):
        dataset.add(EvalCase(f"c{i}", input=i))
    report = await SuiteRunner(app, dataset, [expect_completed()], concurrency=3).arun()
    assert report.passed
    assert max_active <= 3
    assert len(report.results) == 8


def test_suite_runner_rejects_invalid_concurrency() -> None:
    try:
        SuiteRunner(_app, _dataset(), [expect_completed()], concurrency=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for concurrency=0")


def test_failures_carry_full_reports() -> None:
    report = SuiteRunner(_app, _dataset(), [expect_completed()]).run()
    failing = report.failures[0]
    assert failing.case_id == "bad"
    assert failing.run.status is RunStatus.FAILED
    assert not failing.report.passed
    assert failing.report.results[0].message
