"""MCP tool node: call an MCP server tool from inside a graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pyantra.graph.node import NodeConfig
from pyantra.mcp.client import McpClientProtocol
from pyantra.mcp.schema import json_schema_to_python, required_properties
from pyantra.state.state import StateT
from pyantra.tools.base import ArgsFrom, ToolError, ToolNode


class McpToolNode(ToolNode[StateT]):
    """A node that calls a tool on an MCP server.

    The tool's typed ``inputSchema`` is pulled from the server manifest so
    argument types flow into the same compile-time validation the rest of the
    graph uses, and results merge into ``result_field`` through the graph's
    reducers — the MCP path does not bypass the ``Annotated`` type system.

    ``client`` is a (possibly unconnected) :class:`~pyantra.mcp.McpClient`.
    The manifest is fetched on the first invocation. To enable schema-based
    validation without a live connection — e.g. in unit tests — pass
    ``input_schema`` directly; it is also retained after the manifest is
    fetched.
    """

    def __init__(
        self,
        *,
        name: str,
        client: McpClientProtocol,
        tool_name: str,
        result_field: str,
        args_from: ArgsFrom = None,
        input_schema: dict[str, Any] | None = None,
        config: NodeConfig | None = None,
    ) -> None:
        super().__init__(
            name=name, result_field=result_field, args_from=args_from, config=config
        )
        self.client = client
        self.tool_name = tool_name
        if input_schema is not None:
            self._apply_schema(input_schema)

    async def _call_tool(self, arguments: Mapping[str, Any]) -> Any:
        await self._load_schema()
        return await self.client.call_tool(self.tool_name, arguments)

    async def _load_schema(self) -> None:
        """Fetch the tool manifest from the server and cache its schema."""
        if self.input_schema is not None:
            return
        for tool in await self.client.list_tools():
            if tool["name"] == self.tool_name:
                self._apply_schema(tool["inputSchema"] or {})
                return
        raise ToolError(f"MCP server has no tool named {self.tool_name!r}.")

    def _apply_schema(self, schema: dict[str, Any]) -> None:
        self.input_schema = schema
        self.argument_types = json_schema_to_python(schema)
        required = required_properties(schema)
        self.required_args = (
            required if required else set()
        )


__all__ = ["McpToolNode"]
