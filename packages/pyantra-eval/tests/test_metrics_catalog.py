import pytest

from pyantra import MockLLM, Run, RunEvent, RunStatus
from pyantra_eval import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEvalMetric,
    HallucinationMetric,
    TaskCompletionMetric,
    ToolSelectionMetric,
    evaluate,
)


def _event(node: str) -> RunEvent:
    return RunEvent(run_id="r1", event="node", timestamp=1.0, node=node)


def _run(
    status: RunStatus = RunStatus.COMPLETED,
    events: list[RunEvent] | None = None,
    state: object = None,
) -> Run:
    return Run(run_id="r1", status=status, events=events or [], state=state)


def test_task_completion_passes_on_completed_with_output() -> None:
    metric = TaskCompletionMetric()
    result = metric.evaluate(_run(state="done"))
    assert result.passed
    assert result.score == 1.0


def test_task_completion_fails_on_failed_run() -> None:
    result = TaskCompletionMetric().evaluate(_run(RunStatus.FAILED, state="text"))
    assert not result.passed
    assert result.score == 0.0


def test_task_completion_fails_on_empty_output() -> None:
    result = TaskCompletionMetric().evaluate(_run(state=""))
    assert not result.passed
    assert "no output" in result.message


def test_task_completion_respects_extractor() -> None:
    metric = TaskCompletionMetric(extract=lambda run: run.state["answer"])
    run = _run(state={"answer": "ok"})
    assert metric.evaluate(run).passed
    assert not metric.evaluate(_run(state={"answer": ""})).passed


def test_tool_selection_passes_when_trace_matches() -> None:
    metric = ToolSelectionMetric(
        expected_nodes=("fetch", "store"), forbidden_nodes=("delete",)
    )
    run = _run(events=[_event("fetch"), _event("store")])
    result = metric.evaluate(run)
    assert result.passed
    assert result.score == 1.0


def test_tool_selection_scores_partial_match() -> None:
    metric = ToolSelectionMetric(expected_nodes=("a", "b"), forbidden_nodes=("c",))
    run = _run(events=[_event("a"), _event("c")])
    result = metric.evaluate(run)
    assert not result.passed
    assert result.score == pytest.approx(1 / 3)
    assert "missing" in result.message
    assert "unexpected" in result.message


def test_tool_selection_empty_constraints_passes() -> None:
    result = ToolSelectionMetric().evaluate(_run(events=[_event("a")]))
    assert result.passed
    assert result.score == 1.0


def test_tool_selection_threshold() -> None:
    metric = ToolSelectionMetric(
        expected_nodes=("a", "b"), threshold=0.5
    )
    result = metric.evaluate(_run(events=[_event("a")]))
    assert result.score == pytest.approx(0.5)
    assert result.passed


def test_faithfulness_judges_output_against_context() -> None:
    llm = MockLLM(responses=["Score: 9/10\nRationale: Supported."])
    metric = FaithfulnessMetric(llm, max_score=10.0)
    run = _run(
        state="The sky is blue.",
        events=[_event("answer")],
    )
    result = metric.evaluate(run)
    assert result.passed
    assert result.score == 9.0
    assert result.max_score == 10.0
    prompt = llm.recorded_calls[0][0].content
    assert "The sky is blue." in prompt


def test_faithfulness_fails_below_threshold() -> None:
    llm = MockLLM(responses=["Score: 2/10\nRationale: Unsupported."])
    metric = FaithfulnessMetric(llm, max_score=10.0, threshold=5.0)
    result = metric.evaluate(_run(state="Made up fact."))
    assert not result.passed
    assert result.message


def test_faithfulness_extracts_context_from_dict_state() -> None:
    llm = MockLLM(responses=["Score: 8/10\nRationale: Fine."])
    metric = FaithfulnessMetric(
        llm,
        max_score=10.0,
        extract=lambda run: run.state["answer"],
    )
    run = _run(state={"answer": "Response text", "context": "Grounding text"})
    result = metric.evaluate(run)
    prompt = llm.recorded_calls[0][0].content
    assert "Grounding text" in prompt
    assert "Response text" in prompt
    assert result.passed


def test_answer_relevancy_uses_question_and_response() -> None:
    llm = MockLLM(responses=["Score: 7/10\nRationale: On topic."])
    metric = AnswerRelevancyMetric(
        llm,
        max_score=10.0,
        extract=lambda run: run.state["answer"],
        extract_context=lambda run: run.state["question"],
    )
    run = _run(
        state={
            "answer": "Paris is the capital.",
            "question": "What is the capital of France?",
        }
    )
    result = metric.evaluate(run)
    prompt = llm.recorded_calls[0][0].content
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital." in prompt
    assert result.score == 7.0
    assert result.passed


def test_hallucination_judges_contradiction() -> None:
    llm = MockLLM(responses=["Score: 3/10\nRationale: Contradicts context."])
    metric = HallucinationMetric(llm, max_score=10.0)
    result = metric.evaluate(
        _run(
            state="The sky is green.",
            events=[_event("answer")],
        )
    )
    assert not result.passed
    assert result.score == 3.0


def test_geval_scores_single_sample() -> None:
    llm = MockLLM(responses=["Score: 8/10\nRationale: Solid."])
    metric = GEvalMetric(llm, rubric="Coherence", max_score=10.0)
    result = metric.evaluate(_run(state="Good answer."))
    assert result.passed
    assert result.score == 8.0


def test_geval_includes_evaluation_steps_in_prompt() -> None:
    llm = MockLLM(responses=["Score: 8/10\nRationale: Fine."])
    metric = GEvalMetric(
        llm,
        rubric="Coherence",
        max_score=10.0,
        evaluation_steps=("Check clarity.", "Check tone."),
    )
    metric.evaluate(_run(state="text"))
    prompt = llm.recorded_calls[0][0].content
    assert "Evaluation steps" in prompt
    assert "1. Check clarity." in prompt
    assert "2. Check tone." in prompt


def test_geval_averages_samples_and_sums_usage() -> None:
    llm = MockLLM(
        responses=[
            "Score: 6/10\nRationale: A.",
            "Score: 8/10\nRationale: B.",
            "Score: 10/10\nRationale: C.",
        ]
    )
    metric = GEvalMetric(llm, rubric="Quality", max_score=10.0, n_samples=3)
    result = metric.evaluate(_run(state="text"))
    assert result.score == pytest.approx(8.0)
    assert result.passed
    assert result.usage.input_tokens == 30


def test_geval_respects_threshold_on_mean() -> None:
    llm = MockLLM(
        responses=["Score: 6/10\nRationale: A.", "Score: 4/10\nRationale: B."]
    )
    metric = GEvalMetric(llm, rubric="Quality", max_score=10.0, n_samples=2)
    result = metric.evaluate(_run(state="text"))
    assert result.score == pytest.approx(5.0)
    assert result.passed  # default threshold = max_score / 2 = 5.0


def test_metrics_compose_with_evaluate() -> None:
    llm = MockLLM(responses=["Score: 9/10\nRationale: Good."])
    run = _run(
        events=[_event("fetch"), _event("store")],
        state={"answer": "ok", "context": "ctx"},
    )
    report = evaluate(
        run,
        TaskCompletionMetric(extract=lambda r: r.state["answer"]),
        ToolSelectionMetric(expected_nodes=("fetch", "store")),
        FaithfulnessMetric(
            llm, max_score=10.0, extract=lambda r: r.state["answer"]
        ),
    )
    assert report.passed
    assert report.avg_score is not None
