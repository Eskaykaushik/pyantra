from pyantra import MockLLM, Usage
from pyantra_eval import DiskCache, JudgeResult, LLMJudge


def _result() -> JudgeResult:
    return JudgeResult(
        score=7.5,
        rationale="Good.",
        passed=True,
        usage=Usage(input_tokens=10, output_tokens=5, cost=0.01, model="mock"),
        message="",
    )


def test_roundtrip(tmp_path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("abc", _result())
    cached = cache.get("abc")
    assert cached is not None
    assert cached.score == 7.5
    assert cached.rationale == "Good."
    assert cached.passed is True
    assert cached.usage.input_tokens == 10
    assert cached.usage.output_tokens == 5
    assert cached.usage.cost == 0.01
    assert cached.usage.model == "mock"


def test_missing_key_returns_none(tmp_path) -> None:
    assert DiskCache(tmp_path).get("nope") is None


def test_persists_across_instances(tmp_path) -> None:
    DiskCache(tmp_path).set("k", _result())
    assert DiskCache(tmp_path).get("k") is not None


def test_corrupt_file_returns_none(tmp_path) -> None:
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    assert DiskCache(tmp_path).get("bad") is None


def test_judge_serves_from_disk_cache(tmp_path) -> None:
    cache = DiskCache(tmp_path)
    llm = MockLLM(responses=["Score: 9/10\nRationale: Good."])
    LLMJudge(llm, rubric="Quality", max_score=10.0, cache=cache).judge("text")
    assert len(llm.recorded_calls) == 1

    fresh_llm = MockLLM(responses=["Score: 1/10\nRationale: Never used."])
    judge = LLMJudge(
        fresh_llm, rubric="Quality", max_score=10.0, cache=DiskCache(tmp_path)
    )
    result = judge.judge("text")
    assert result.score == 9.0
    assert len(fresh_llm.recorded_calls) == 0
    assert result.usage.total_tokens == 0
