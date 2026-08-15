from pyantra import MockLLM
from pyantra_eval import JudgeResult, LLMJudge


def test_judge_parses_integer_score() -> None:
    llm = MockLLM(responses=["Score: 8\nRationale: Clear and correct."])
    judge = LLMJudge(llm, rubric="Answer quality", max_score=10.0)
    result = judge.judge("the answer")
    assert result.score == 8.0
    assert result.passed is True
    assert result.rationale == "Clear and correct."


def test_judge_parses_fraction_score() -> None:
    llm = MockLLM(responses=["Score: 7/10\nRationale: Mostly right."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    assert judge.judge("x").score == 7.0


def test_judge_normalizes_fraction_to_max_score() -> None:
    llm = MockLLM(responses=["Score: 7/10\nRationale: Solid."])
    judge = LLMJudge(llm, rubric="Quality", max_score=5.0)
    result = judge.judge("x")
    assert result.score == 3.5


def test_judge_bare_number_is_absolute_on_scale() -> None:
    llm = MockLLM(responses=["Score: 0.7\nRationale: Fine."])
    judge = LLMJudge(llm, rubric="Quality", max_score=5.0)
    assert judge.judge("x").score == 0.7


def test_judge_clamps_out_of_range_scores() -> None:
    llm = MockLLM(responses=["Score: 15\nRationale: Overstated."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    assert judge.judge("x").score == 10.0


def test_judge_threshold_controls_passed() -> None:
    llm = MockLLM(responses=["Score: 3\nRationale: So-so."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    assert judge.judge("x", threshold=2.0).passed is True
    assert judge.judge("x", threshold=5.0).passed is False


def test_judge_default_threshold_is_half() -> None:
    llm = MockLLM(responses=["Score: 4\nRationale: OK."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    assert judge.judge("x").passed is False


def test_judge_missing_score_reports_failure() -> None:
    llm = MockLLM(responses=["I cannot score this."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    result = judge.judge("x")
    assert not result.passed
    assert result.score == 0.0
    assert result.message


def test_judge_records_usage() -> None:
    llm = MockLLM(
        responses=["Score: 9\nRationale: Great."],
        input_tokens=50,
        output_tokens=12,
        cost=0.01,
    )
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    result = judge.judge("x")
    assert isinstance(result, JudgeResult)
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 12
    assert result.usage.cost == 0.01


async def test_ajudge_parses_async() -> None:
    llm = MockLLM(responses=["Score: 6/10\nRationale: Decent."])
    judge = LLMJudge(llm, rubric="Quality", max_score=10.0)
    result = await judge.ajudge("x")
    assert result.score == 6.0
    assert result.passed is True


def test_judge_prompts_include_rubric_and_input() -> None:
    llm = MockLLM(responses=["Score: 5\nRationale: Fine."])
    judge = LLMJudge(llm, rubric="Be concise", max_score=10.0)
    judge.judge("hello world")
    prompt = llm.recorded_calls[0][0].content
    assert "Be concise" in prompt
    assert "hello world" in prompt
    assert "0 to 10" in prompt
