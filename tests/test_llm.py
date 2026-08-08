"""Tests for the LLM abstraction: types, MockLLM, and usage tracking."""

from __future__ import annotations

import pytest
from conftest import State

from pyantra import (
    END,
    LLMResponse,
    Message,
    MockLLM,
    Usage,
    UsageTracker,
)
from pyantra.graph.graph import Graph


def test_message_construction() -> None:
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_usage_total_tokens() -> None:
    usage = Usage(input_tokens=10, output_tokens=5, cache_tokens=3)
    assert usage.total_tokens == 18


def test_usage_addition_aggregates() -> None:
    a = Usage(input_tokens=10, output_tokens=5, cost=0.2, model="gpt-4o-mini")
    b = Usage(input_tokens=2, output_tokens=3, cost=0.1, model="gpt-4o-mini")
    total = a + b
    assert total.input_tokens == 12
    assert total.output_tokens == 8
    assert total.total_tokens == 20
    assert total.cost == pytest.approx(0.3)
    assert total.model == "gpt-4o-mini"


def test_mock_llm_returns_scripted_responses_in_order() -> None:
    llm = MockLLM(responses=["first", "second"], input_tokens=7, output_tokens=3)
    resp1 = llm.generate([Message(role="user", content="q1")])
    resp2 = llm.generate([Message(role="user", content="q2")])
    assert isinstance(resp1, LLMResponse)
    assert resp1.content == "first"
    assert resp2.content == "second"
    assert resp1.usage.input_tokens == 7
    assert resp1.usage.output_tokens == 3
    assert resp1.usage.total_tokens == 10


def test_mock_llm_cycles_when_responses_exhausted() -> None:
    llm = MockLLM(responses=["only"])
    assert llm.generate([]).content == "only"
    assert llm.generate([]).content == "only"


def test_mock_llm_records_calls() -> None:
    llm = MockLLM(responses=["ok"])
    msgs = [Message(role="user", content="q")]
    llm.generate(msgs)
    assert llm.recorded_calls == [msgs]


def test_mock_llm_empty_responses_raise() -> None:
    llm = MockLLM()
    with pytest.raises(ValueError, match="no scripted responses"):
        llm.generate([])


@pytest.mark.asyncio
async def test_mock_llm_agenerate_matches_generate() -> None:
    llm = MockLLM(responses=["async"])
    resp = await llm.agenerate([Message(role="user", content="q")])
    assert resp.content == "async"
    assert len(llm.recorded_calls) == 1


def test_usage_tracker_accumulates_and_resets() -> None:
    tracker = UsageTracker()
    assert tracker.total == Usage()
    tracker.add(Usage(input_tokens=10, output_tokens=5, cost=0.1, model="mock"))
    tracker.add(Usage(input_tokens=2, output_tokens=1, cost=0.05, model="mock"))
    total = tracker.total
    assert total.input_tokens == 12
    assert total.output_tokens == 6
    assert total.cost == pytest.approx(0.15)
    tracker.reset()
    assert tracker.total == Usage()


def test_tracker_records_multiple_mock_calls() -> None:
    tracker = UsageTracker()
    llm = MockLLM(responses=["a", "b"], input_tokens=4, output_tokens=2)
    for _ in range(3):
        tracker.add(llm.generate([]).usage)
    assert tracker.total.input_tokens == 12
    assert tracker.total.output_tokens == 6


def test_graph_run_with_llm_and_tracker(graph: Graph[State]) -> None:
    tracker = UsageTracker()
    llm = MockLLM(responses=["summarized"], input_tokens=3, output_tokens=2)

    def summarize(state: State) -> State:
        resp = llm.generate([Message(role="user", content=str(state.value))])
        tracker.add(resp.usage)
        state.history.append(resp.content)
        return state

    graph.add_node(summarize, name="summarize")
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", END)
    app = graph.compile()

    run = app.run(State(value=1))
    assert run.status.value == "completed"
    assert run.state is not None
    assert run.state.history == ["summarized"]
    assert tracker.total.input_tokens == 3
    assert tracker.total.output_tokens == 2
    assert len(llm.recorded_calls) == 1
