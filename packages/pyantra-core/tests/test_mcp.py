"""Tests for MCP tool nodes and JSON Schema mapping."""

from __future__ import annotations

import operator
import sys
from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest

from pyantra import (
    END,
    Graph,
    GraphCompileError,
    McpClient,
    McpToolNode,
    NodeExecutionError,
    RunStatus,
    ToolError,
    json_schema_to_python,
)
from pyantra.mcp.schema import required_properties


@dataclass
class McpState:
    query: str = ""
    n: int = 0
    results: Annotated[list[str], operator.add] = field(default_factory=list)


class FakeClient:
    def __init__(self, tools=None, result=None, error=None) -> None:
        self.tools = list(tools or [])
        self.result = result
        self.error = error
        self.calls = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        if self.error:
            raise ToolError(self.error)
        return self.result


def _echo_tool() -> FakeClient:
    return FakeClient(
        tools=[
            {
                "name": "echo",
                "inputSchema": {
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                },
            }
        ],
        result=["hello"],
    )


def test_json_schema_to_python() -> None:
    schema = {
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "items": {"type": "array"},
            "meta": {"type": "object"},
            "nil": {"type": "null"},
            "untyped": {},
            "either": {"type": ["string", "null"]},
            "mixed": {"type": ["string", "integer"]},
        }
    }
    types = json_schema_to_python(schema)
    assert types["name"] is str
    assert types["count"] is int
    assert types["ratio"] is float
    assert types["flag"] is bool
    assert types["items"] is list
    assert types["meta"] is dict
    assert types["nil"] is type(None)
    assert types["untyped"] is Any
    assert types["either"] is str
    assert types["mixed"] is Any


def test_required_properties() -> None:
    assert required_properties({"required": ["name", "count"]}) == {"name", "count"}
    assert required_properties({}) == set()


def test_mcp_tool_node_calls_tool_and_merges() -> None:
    client = _echo_tool()
    graph = Graph(McpState)
    tool = McpToolNode(
        name="echo",
        client=client,
        tool_name="echo",
        result_field="results",
        args_from={"query": "text"},
        input_schema={
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(McpState(query="hi"))
    assert run.status is RunStatus.COMPLETED
    assert client.calls == [("echo", {"text": "hi"})]
    assert run.state.results == ["hello"]


def test_mcp_tool_node_loads_schema_lazily() -> None:
    client = _echo_tool()
    graph = Graph(McpState)
    tool = McpToolNode(
        name="echo",
        client=client,
        tool_name="echo",
        result_field="results",
        args_from={"query": "text"},
    )
    assert tool.input_schema is None
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(McpState(query="hi"))
    assert run.status is RunStatus.COMPLETED
    assert tool.input_schema is not None
    assert tool.argument_types == {"text": str}
    assert tool.required_args == {"text"}


def test_mcp_tool_node_unknown_tool_fails() -> None:
    client = FakeClient(tools=[])
    graph = Graph(McpState)
    tool = McpToolNode(
        name="x", client=client, tool_name="nope", result_field="results"
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(McpState())
    assert run.status is RunStatus.FAILED
    assert isinstance(run.exception, NodeExecutionError)
    assert isinstance(run.exception.__cause__, ToolError)


def test_mcp_tool_node_client_error_fails() -> None:
    client = FakeClient(error="boom")
    graph = Graph(McpState)
    tool = McpToolNode(
        name="x", client=client, tool_name="echo", result_field="results"
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(McpState())
    assert run.status is RunStatus.FAILED
    assert isinstance(run.exception, NodeExecutionError)
    assert isinstance(run.exception.__cause__, ToolError)


def test_mcp_tool_node_argument_type_mismatch_fails() -> None:
    client = _echo_tool()
    graph = Graph(McpState)
    tool = McpToolNode(
        name="echo",
        client=client,
        tool_name="echo",
        result_field="results",
        args_from={"n": "text"},
        input_schema={"properties": {"text": {"type": "string"}}},
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)

    run = graph.compile().run(McpState(n=7))
    assert run.status is RunStatus.FAILED
    assert isinstance(run.exception, NodeExecutionError)
    assert isinstance(run.exception.__cause__, ToolError)


def test_mcp_tool_node_unknown_result_field_fails_compile() -> None:
    graph = Graph(McpState)
    tool = McpToolNode(
        name="echo", client=_echo_tool(), tool_name="echo", result_field="bogus"
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)
    with pytest.raises(GraphCompileError, match="bogus"):
        graph.compile()


def test_mcp_tool_node_missing_required_input_fails_compile() -> None:
    graph = Graph(McpState)
    tool = McpToolNode(
        name="echo",
        client=_echo_tool(),
        tool_name="echo",
        result_field="results",
        args_from={"query": "different"},
        input_schema={
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
    )
    graph.add_node(tool)
    graph.set_entry_point(tool)
    graph.add_edge(tool, END)
    with pytest.raises(GraphCompileError, match="text"):
        graph.compile()


def test_mcp_imports_do_not_pull_in_sdk() -> None:
    import pyantra.mcp  # noqa: F401

    assert "mcp" not in sys.modules


def test_mcp_client_requires_exactly_one_transport() -> None:
    with pytest.raises(ValueError):
        McpClient()
    with pytest.raises(ValueError):
        McpClient(command="npx", url="http://localhost:8000")
    client = McpClient(command="npx")
    assert client.connected is False
