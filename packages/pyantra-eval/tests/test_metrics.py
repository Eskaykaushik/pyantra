from pyantra import Run, RunEvent, RunStatus
from pyantra_eval import (
    EvalReport,
    evaluate,
    expect_completed,
    expect_failed,
    expect_interrupt,
    expect_judged,
    expect_max_steps,
    expect_not_visited,
    expect_ordered,
    expect_status,
    expect_that,
    expect_visited,
)


def _event(node: str | None = None, message: str | None = None) -> RunEvent:
    return RunEvent(
        run_id="r1", event="node", timestamp=1.0, node=node, message=message
    )


def _run(
    status: RunStatus = RunStatus.COMPLETED,
    events: list[RunEvent] | None = None,
    error: str | None = None,
    state: object = None,
) -> Run:
    return Run(
        run_id="r1",
        status=status,
        events=events or [],
        error=error,
        state=state,
    )


def test_expect_status() -> None:
    assert expect_status(RunStatus.COMPLETED).evaluate(_run()).passed
    result = expect_status(RunStatus.COMPLETED).evaluate(_run(RunStatus.FAILED))
    assert not result.passed
    assert "completed" in result.message


def test_expect_completed_and_failed() -> None:
    assert expect_completed().evaluate(_run()).passed
    assert expect_failed().evaluate(_run(RunStatus.FAILED)).passed
    assert not expect_failed().evaluate(_run()).passed


def test_expect_error_contains_and_pattern() -> None:
    run = _run(RunStatus.FAILED, error="boom: model timed out")
    assert expect_failed(contains="timed out").evaluate(run).passed
    assert not expect_failed(contains="nope").evaluate(run).passed
    assert expect_failed(pattern=r"model \w+ out").evaluate(run).passed
    assert not expect_failed(pattern=r"model \d+ out").evaluate(run).passed


def test_expect_visited_and_at_least() -> None:
    run = _run(events=[_event("a"), _event("b"), _event("a")])
    assert expect_visited("a").evaluate(run).passed
    assert expect_visited("a", at_least=2).evaluate(run).passed
    assert not expect_visited("a", at_least=3).evaluate(run).passed
    assert not expect_visited("z").evaluate(run).passed


def test_expect_not_visited() -> None:
    run = _run(events=[_event("a"), _event("b")])
    assert expect_not_visited("z").evaluate(run).passed
    assert not expect_not_visited("a").evaluate(run).passed


def test_expect_ordered() -> None:
    run = _run(events=[_event("a"), _event("b"), _event("c")])
    assert expect_ordered("a", "b", "c").evaluate(run).passed
    assert not expect_ordered("a", "c", "b").evaluate(run).passed
    assert not expect_ordered("a", "b", "z").evaluate(run).passed
    assert expect_ordered("a", "c").evaluate(run).passed


def test_expect_max_steps() -> None:
    run = _run(events=[_event("a"), _event("b")])
    assert expect_max_steps(2).evaluate(run).passed
    assert not expect_max_steps(1).evaluate(run).passed


def test_expect_interrupt() -> None:
    assert expect_interrupt().evaluate(_run(RunStatus.PAUSED)).passed
    assert not expect_interrupt().evaluate(_run()).passed


def test_expect_that() -> None:
    expectation = expect_that(
        lambda run: run.state == "done",
        message="state must be 'done'",
        name="state_done",
    )
    assert expectation.evaluate(_run(state="done")).passed
    result = expectation.evaluate(_run(state="pending"))
    assert not result.passed
    assert result.message == "state must be 'done'"


def test_evaluate_aggregates_and_failures() -> None:
    run = _run(events=[_event("a")])
    report = evaluate(
        run, expect_completed(), expect_visited("a"), expect_visited("z")
    )
    assert isinstance(report, EvalReport)
    assert not report.passed
    assert [r.name for r in report.failures] == ["visited"]
    assert report.to_dict()["passed"] is False


def test_evaluate_all_pass() -> None:
    run = _run(events=[_event("a")], state="done")
    report = evaluate(
        run, expect_completed(), expect_visited("a"), expect_ordered("a")
    )
    assert report.passed
    assert report.failures == []


def test_expect_judged_uses_state_by_default() -> None:
    from pyantra import MockLLM
    from pyantra_eval import LLMJudge

    llm = MockLLM(responses=["Score: 0.9\nRationale: Solid."])
    judge = LLMJudge(llm, rubric="Quality", max_score=1.0)
    run = _run(state="good answer")
    assert expect_judged(judge).evaluate(run).passed


def test_expect_judged_with_extractor_and_threshold() -> None:
    from pyantra import MockLLM
    from pyantra_eval import LLMJudge

    llm = MockLLM(responses=["Score: 2\nRationale: Weak."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    run = _run(state={"summary": "meh"})
    result = expect_judged(judge, extract=lambda r: r.state["summary"], threshold=5.0)
    assert not result.evaluate(run).passed
    assert "score 2.00" in result.evaluate(run).message
