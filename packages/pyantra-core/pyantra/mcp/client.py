"""Minimal MCP client adapter.

The official ``mcp`` package is imported lazily so pyantra core stays
dependency-free unless MCP is actually used. Install it with
``pip install 'pyantra[mcp]'``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pyantra.runtime.errors import PyantraError
from pyantra.tools.base import ToolError


class McpClientProtocol(Protocol):
    """The interface ``McpToolNode`` relies on.

    ``list_tools`` returns normalized manifests ``{"name", "inputSchema"}``
    and ``call_tool`` returns a plain value (structured content when the
    server provides it, otherwise the concatenated text content).
    """

    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> Any: ...


@dataclass(frozen=True)
class _McpModules:
    ClientSession: type[Any]
    StdioServerParameters: type[Any]
    stdio: Any
    streamable_http: Any


_MCP_MODULES: _McpModules | None = None


def _import_mcp() -> _McpModules:
    """Import the official MCP SDK, raising a clear error if absent."""
    global _MCP_MODULES
    if _MCP_MODULES is not None:
        return _MCP_MODULES
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client import stdio, streamable_http
    except ImportError as exc:
        raise PyantraError(
            "MCP support requires the 'mcp' package. Install it with "
            "pip install 'pyantra[mcp]'."
        ) from exc
    _MCP_MODULES = _McpModules(
        ClientSession=ClientSession,
        StdioServerParameters=StdioServerParameters,
        stdio=stdio,
        streamable_http=streamable_http,
    )
    return _MCP_MODULES


class McpClient:
    """A client for an MCP server over stdio or streamable HTTP.

    Example::

        async with McpClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ) as client:
            tool = McpToolNode(
                name="read",
                client=client,
                tool_name="read_file",
                result_field="contents",
            )

    Both ``command`` (stdio) and ``url`` (streamable HTTP) may be given, but
    not both. The connection is established lazily on the first tool call (or
    by :meth:`connect`).
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if (command is None) == (url is None):
            raise ValueError(
                "McpClient requires exactly one of command (stdio) or url "
                "(streamable HTTP)."
            )
        self._command = command
        self._args = list(args or [])
        self._url = url
        self._headers = dict(headers or {})
        self._session: Any = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._tools: dict[str, dict[str, Any]] = {}

    @property
    def connected(self) -> bool:
        """Whether the underlying MCP session is open."""
        return self._session is not None

    async def connect(self) -> McpClient:
        """Start the server process (stdio) or connect to the URL."""
        modules = _import_mcp()
        stack = contextlib.AsyncExitStack()
        if self._url is not None:
            ctx = modules.streamable_http.streamablehttp_client(
                self._url, headers=self._headers
            )
        else:
            params = modules.StdioServerParameters(
                command=self._command, args=self._args
            )
            ctx = modules.stdio.stdio_client(params)
        read, write = await stack.enter_async_context(ctx)
        session = await stack.enter_async_context(
            modules.ClientSession(read, write)
        )
        await session.initialize()
        self._session = session
        self._stack = stack
        self._tools = {}
        return self

    async def close(self) -> None:
        """Shut down the connection and release the server process."""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._tools = {}

    async def __aenter__(self) -> McpClient:
        return self if self._session is not None else await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return normalized tool manifests (``name`` + ``inputSchema``)."""
        await self._ensure_session()
        if not self._tools:
            result = await self._session.list_tools()
            self._tools = {
                tool.name: {"name": tool.name, "inputSchema": tool.inputSchema or {}}
                for tool in result.tools
            }
        return list(self._tools.values())

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> Any:
        """Call a tool and return its result as a plain value.

        Structured content is returned when the server provides it; otherwise
        the tool's text content blocks are joined. A tool that reports an
        error raises :class:`~pyantra.tools.ToolError`.
        """
        await self._ensure_session()
        result = await self._session.call_tool(name, arguments=dict(arguments))
        if getattr(result, "isError", False):
            raise ToolError(f"MCP tool {name!r} reported an error: {result}")
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        content = getattr(result, "content", None) or []
        texts = [item.text for item in content if hasattr(item, "text")]
        if len(texts) == 1:
            return texts[0]
        if texts:
            return "\n".join(texts)
        return result

    async def _ensure_session(self) -> None:
        if self._session is None:
            await self.connect()


__all__ = ["McpClient", "McpClientProtocol"]
