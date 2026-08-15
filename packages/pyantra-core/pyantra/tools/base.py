"""Nodes that call external tools and merge results through reducers.

Tool nodes are ordinary :class:`~pyantra.graph.node.Node` instances: they
receive state, return a partial update, and their results merge through the
graph's ``Annotated`` reducers exactly like any other node. What distinguishes
them is the typed schema — pulled from an MCP tool manifest
(:class:`~pyantra.mcp.McpToolNode`) or a plain callable's signature
(:class:`FunctionTool`) — which feeds compile-time validation against the
state type instead of bypassing it.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias, get_type_hints

from pyantra.graph.node import Node, NodeConfig
from pyantra.runtime.errors import GraphCompileError, PyantraError
from pyantra.state.state import StateT, StateUpdate

ArgsFrom: TypeAlias = Callable[[Any], Mapping[str, Any]] | Mapping[str, str] | None


class ToolError(PyantraError):
    """Raised when a tool call fails or the tool reports an error result."""


class ToolNode(Node[StateT]):
    """A node that calls an external tool and merges its result into state.

    Tool arguments are extracted from state via ``args_from``:

    * a callable ``(state) -> dict`` returning the tool arguments directly;
    * a mapping ``{state_field: tool_arg_name}`` pulling named state fields;
    * ``None`` — the tool is called with no arguments.

    The tool result is written to ``result_field`` as a partial state update,
    so it merges through the graph's reducers (e.g. an
    ``Annotated[list, add]`` field) like any other node return.
    """

    def __init__(
        self,
        *,
        name: str,
        result_field: str,
        args_from: ArgsFrom = None,
        config: NodeConfig | None = None,
    ) -> None:
        super().__init__(name, self._invoke, config)
        self.result_field = result_field
        self.args_from = args_from
        # Populated by subclasses that carry a schema (manifest or signature).
        self.input_schema: dict[str, Any] | None = None
        self.argument_types: dict[str, type[Any]] | None = None
        self.required_args: set[str] | None = None

    def _call_tool(self, arguments: Mapping[str, Any]) -> Any | Awaitable[Any]:
        """Invoke the underlying tool. Implemented by subclasses."""
        raise NotImplementedError

    def _invoke(self, state: StateT) -> StateUpdate | Awaitable[StateUpdate]:
        arguments = self._extract_args(state)
        if self.argument_types:
            _validate_arguments(arguments, self.argument_types)
        result = self._call_tool(arguments)
        if inspect.isawaitable(result):

            async def awaited() -> StateUpdate:
                return {self.result_field: await result}

            return awaited()
        return {self.result_field: result}

    def _extract_args(self, state: StateT) -> dict[str, Any]:
        args_from = self.args_from
        if args_from is None:
            return {}
        if callable(args_from):
            return dict(args_from(state))
        return {
            arg_name: getattr(state, state_field)
            for state_field, arg_name in args_from.items()
        }

    def validate(self, state_type: type[Any]) -> None:
        """Compile-time validation against the state type.

        Requires ``result_field`` and (for mapping-based ``args_from``) every
        referenced state field to exist on the state dataclass. When a typed
        schema is available (``argument_types``), the tool's required inputs
        are checked against the extractable arguments.
        """
        fields = _field_names(state_type)
        if fields is not None and self.result_field not in fields:
            raise GraphCompileError(
                f"Tool node {self.name!r} writes to unknown state field "
                f"{self.result_field!r}."
            )
        if isinstance(self.args_from, Mapping):
            for state_field in self.args_from:
                if fields is not None and state_field not in fields:
                    raise GraphCompileError(
                        f"Tool node {self.name!r} reads unknown state field "
                        f"{state_field!r}."
                    )
        if isinstance(self.args_from, Mapping) and self.argument_types:
            expected = (
                self.required_args
                if self.required_args is not None
                else set(self.argument_types)
            )
            provided = set(self.args_from.values())
            missing = expected - provided
            if missing:
                raise GraphCompileError(
                    f"Tool node {self.name!r} does not provide arguments for "
                    f"tool inputs: {', '.join(sorted(missing))}."
                )


class FunctionTool(ToolNode[StateT]):
    """A tool node backed by a plain Python callable.

    The callable is invoked as ``fn(**arguments)``. Argument types are derived
    from the callable's annotated signature so the direct-call path gets the
    same compile-time checks an MCP manifest provides — calling a plain
    function and calling an MCP tool are equivalent in how they merge state.
    """

    def __init__(
        self,
        *,
        name: str,
        fn: Callable[..., Any],
        result_field: str,
        args_from: ArgsFrom = None,
        config: NodeConfig | None = None,
    ) -> None:
        super().__init__(
            name=name, result_field=result_field, args_from=args_from, config=config
        )
        self.func = fn
        hints = _signature_hints(fn)
        if hints:
            self.argument_types = {
                key: value for key, value in hints.items() if key != "return"
            }
            required = _required_params(fn)
            self.required_args = {
                name for name in required if name in self.argument_types
            }

    def _call_tool(self, arguments: Mapping[str, Any]) -> Any | Awaitable[Any]:
        return self.func(**dict(arguments))


def _validate_arguments(
    arguments: Mapping[str, Any], types: Mapping[str, type[Any]]
) -> None:
    """Type-check ``arguments`` against ``types``, raising ``ToolError``."""
    for name, expected in types.items():
        value = arguments.get(name)
        if value is None or expected is Any or expected is object:
            continue
        if _isinstance(value, expected):
            continue
        raise ToolError(
            f"Argument {name!r} expected {_type_name(expected)}, got "
            f"{type(value).__name__!r}."
        )


def _isinstance(value: Any, expected: type[Any]) -> bool:
    if expected is int and isinstance(value, bool):
        return False
    if expected is float and isinstance(value, (int, float)) and not isinstance(
        value, bool
    ):
        return True
    try:
        return isinstance(value, expected)
    except TypeError:
        return False


def _type_name(expected: type[Any]) -> str:
    return getattr(expected, "__name__", str(expected))


def _signature_hints(fn: Callable[..., Any]) -> dict[str, type[Any]] | None:
    try:
        hints = get_type_hints(fn)
    except Exception:
        return None
    return hints or None


def _required_params(fn: Callable[..., Any]) -> set[str]:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return set()
    return {
        p.name
        for p in parameters
        if p.default is p.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _field_names(state_type: type[Any]) -> frozenset[str] | None:
    if dataclasses.is_dataclass(state_type):
        return frozenset(f.name for f in dataclasses.fields(state_type))
    return None


__all__ = [
    "ArgsFrom",
    "FunctionTool",
    "ToolError",
    "ToolNode",
    "_validate_arguments",
]
