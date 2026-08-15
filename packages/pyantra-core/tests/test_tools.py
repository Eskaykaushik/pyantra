"""Tests for tool nodes (FunctionTool direct-call path)."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated

import pytest

from pyantra import (
    END,
    FunctionTool,
    Graph,
    GraphCompileError,
    NodeExecutionError,
    RunStatus,
    ToolError,
)


@dataclass
class ToolState:
    query: str = ""
    n: int = 0
    results: Annotated[list[str], operator.add] = field(default_factory=list)


def _echo(text: str) -> list[str]:
    return [text]


def _double(x: int) -> int:
    return x * 2


async def _echo_async(text: str) -> list[str]:
    return [text.upper()]


def test_function_tool_merges_via_reducer() -> None:
    graph = Graph(ToolState)
    first = FunctionTool(
        name="echo-a", fn=_echo, result_field="results", args_from={"query": "text"}
    )
    second = FunctionTool(
        name="echo-b",
        fn=_echo,
        result_field="results",
        args_from=lambda state: {"text": "b"},
    )
    graph.add_node(first)
    graph.add_node(second)
    graph.set_entry_point(first)
    graph.add_edge(first, second)
    graph.add_edge(second, END)

    run = graph.compile().run(ToolState(query="hello"))
    assert run.status is RunStatus.COMPLETED
    assert run.state.results == ["hello", "b"]


def test_function_tool_args_from_callable() -> None:
    graph = Graph(ToolState)
    tool = FunctionTool(
        name="double",
        fn=_double,
        result_field="n",
        args_from=lambda state: {"x": state.n},
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(ToolState(n=21))
    assert run.status is RunStatus.COMPLETED
    assert run.state.n == 42


def test_function_tool_async_callable() -> None:
    graph = Graph(ToolState)
    tool = FunctionTool(
        name="echo", fn=_echo_async, result_field="results", args_from={"query": "text"}
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(ToolState(query="hi"))
    assert run.status is RunStatus.COMPLETED
    assert run.state.results == ["HI"]


def test_function_tool_unknown_result_field_fails_compile() -> None:
    graph = Graph(ToolState)
    tool = FunctionTool(name="echo", fn=_echo, result_field="bogus")
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)
    with pytest.raises(GraphCompileError, match="bogus"):
        graph.compile()


def test_function_tool_unknown_state_field_fails_compile() -> None:
    graph = Graph(ToolState)
    tool = FunctionTool(
        name="echo", fn=_echo, result_field="results", args_from={"bogus": "text"}
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)
    with pytest.raises(GraphCompileError, match="bogus"):
        graph.compile()


def test_function_tool_missing_required_input_fails_compile() -> None:
    graph = Graph(ToolState)
    tool = FunctionTool(
        name="echo",
        fn=_echo,
        result_field="results",
        args_from={"query": "different"},
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)
    with pytest.raises(GraphCompileError, match="text"):
        graph.compile()


def test_function_tool_optional_input_not_required() -> None:
    def greet(name: str, prefix: str = "hi") -> list[str]:
        return [f"{prefix} {name}"]

    graph = Graph(ToolState)
    tool = FunctionTool(
        name="greet", fn=greet, result_field="results", args_from={"query": "name"}
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)
    graph.compile()

    run = graph.compile().run(ToolState(query="ada"))
    assert run.state.results == ["hi ada"]


def test_function_tool_argument_type_mismatch_fails() -> None:
    graph = Graph(ToolState)
    tool = FunctionTool(
        name="double", fn=_double, result_field="n", args_from={"query": "x"}
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(ToolState(query="abc"))
    assert run.status is RunStatus.FAILED
    assert isinstance(run.exception, NodeExecutionError)
    assert isinstance(run.exception.__cause__, ToolError)


def test_add_node_accepts_node_instance() -> None:
    graph = Graph(ToolState)
    tool = FunctionTool(name="echo", fn=_echo, result_field="results")
    registered = graph.add_node(tool)
    assert registered is tool
    assert graph.nodes["echo"] is tool
