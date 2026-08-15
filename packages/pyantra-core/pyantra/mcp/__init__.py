"""MCP support: call Model Context Protocol tools from graphs.

The ``mcp`` package is imported lazily, so pyantra core remains
dependency-free unless MCP is actually used. Install with
``pip install 'pyantra[mcp]'``.
"""

from pyantra.mcp.client import McpClient, McpClientProtocol
from pyantra.mcp.schema import json_schema_to_python, required_properties
from pyantra.mcp.tool_node import McpToolNode

__all__ = [
    "McpClient",
    "McpClientProtocol",
    "McpToolNode",
    "json_schema_to_python",
    "required_properties",
]
