"""Graph: the primary user-facing API for building workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Generic, TypeAlias, overload

from pyantra.graph.compiler import CompiledGraph
from pyantra.graph.conditional import ConditionalEdge, RouterFn
from pyantra.graph.edge import END, Edge, _End
from pyantra.graph.node import Node, NodeConfig, NodeFn
from pyantra.graph.parallel import ParallelEdge
from pyantra.runtime.errors import GraphCompileError
from pyantra.state.reducers import Reducer, extract_reducers
from pyantra.state.state import StateT

NodeLike: TypeAlias = Node[StateT] | str
NodeTarget: TypeAlias = Node[StateT] | str | _End
NodeOrFn: TypeAlias = Node[StateT] | NodeFn[StateT]


class Graph(Generic[StateT]):
    """A typed workflow graph of nodes and edges.

    Example::

        @dataclass
        class State:
            value: int

        graph = Graph(State)

        @graph.node
        def increment(state: State):
            state.value += 1
            return state

        graph.set_entry_point(increment)

        app = graph.compile()
        result = app.run(State(value=1))
    """

    def __init__(self, state_type: type[StateT]) -> None:
        self._state_type = state_type
        self._nodes: dict[str, Node[StateT]] = {}
        self._edges: list[Edge] = []
        self._conditional_edges: list[ConditionalEdge[StateT]] = []
        self._parallel_edges: list[ParallelEdge] = []
        self._entry_point: str | None = None
        self._reducers = extract_reducers(state_type)

    @property
    def state_type(self) -> type[StateT]:
        return self._state_type

    @property
    def reducers(self) -> dict[str, Reducer]:
        """Field reducers extracted from ``Annotated`` metadata on the state type."""
        return dict(self._reducers)

    @property
    def entry_point(self) -> str | None:
        return self._entry_point

    @property
    def nodes(self) -> Mapping[str, Node[StateT]]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    @property
    def conditional_edges(self) -> list[ConditionalEdge[StateT]]:
        return list(self._conditional_edges)

    @property
    def parallel_edges(self) -> list[ParallelEdge]:
        return list(self._parallel_edges)

    def add_node(
        self,
        fn: NodeOrFn[StateT],
        *,
        name: str | None = None,
        config: NodeConfig | None = None,
    ) -> Node[StateT]:
        """Register a function or ``Node`` as a node and return it.

        Passing an existing ``Node`` (e.g. a tool node) registers that node
        object directly, keeping its identity and validation hooks.
        """
        return self._register(fn, name=name, config=config)

    @overload
    def node(
        self,
        fn: NodeOrFn[StateT],
        *,
        name: str | None = None,
        config: NodeConfig | None = None,
    ) -> Node[StateT]: ...

    @overload
    def node(
        self,
        fn: None = None,
        *,
        name: str | None = None,
        config: NodeConfig | None = None,
    ) -> Callable[[NodeOrFn[StateT]], Node[StateT]]: ...

    def node(
        self,
        fn: NodeOrFn[StateT] | None = None,
        *,
        name: str | None = None,
        config: NodeConfig | None = None,
    ) -> Node[StateT] | Callable[[NodeOrFn[StateT]], Node[StateT]]:
        """Decorator to register a function (or ``Node``) as a node.

        Usable as ``@graph.node`` or ``@graph.node(name="...", config=...)``.
        """
        if fn is None:
            return lambda func: self._register(func, name=name, config=config)
        return self._register(fn, name=name, config=config)

    def _register(
        self,
        fn: NodeOrFn[StateT],
        *,
        name: str | None = None,
        config: NodeConfig | None = None,
    ) -> Node[StateT]:
        if isinstance(fn, Node):
            if name is not None and name != fn.name:
                raise GraphCompileError(
                    f"Registered name {name!r} does not match node name "
                    f"{fn.name!r}."
                )
            node = fn
            if config is not None:
                node.config = config
            if node.name in self._nodes:
                raise GraphCompileError(f"Duplicate node name: {node.name!r}.")
            self._nodes[node.name] = node
            return node
        node_name = name if name is not None else getattr(fn, "__name__", None)
        if node_name is None:
            raise GraphCompileError(
                "Cannot infer a node name; provide one via name=... "
                "for callables without __name__."
            )
        if node_name in self._nodes:
            raise GraphCompileError(f"Duplicate node name: {node_name!r}.")
        node = Node(node_name, fn, config=config)
        self._nodes[node_name] = node
        return node

    def set_entry_point(self, node: NodeLike[StateT]) -> None:
        """Set the node where execution begins."""
        self._entry_point = self._resolve(node)

    def add_edge(self, source: NodeLike[StateT], target: NodeTarget[StateT]) -> None:
        """Add an unconditional edge from ``source`` to ``target``.

        ``target`` may be a node or :data:`~pyantra.graph.edge.END`.
        """
        resolved_target = self._resolve_target(target)
        self._edges.append(Edge(self._resolve(source), resolved_target))

    def add_conditional_edges(
        self,
        source: NodeLike[StateT],
        router: RouterFn[StateT],
        path_map: Mapping[str, NodeLike[StateT]] | None = None,
        *,
        default: NodeLike[StateT] | None = None,
    ) -> None:
        """Route execution from ``source`` based on ``router(state)``.

        If ``path_map`` is provided, the router returns a key into it (with an
        optional ``default`` for unknown keys). Otherwise the router returns a
        registered node name directly.
        """
        resolved_path_map = None
        if path_map is not None:
            resolved_path_map = {
                key: self._resolve(target) for key, target in path_map.items()
            }
        resolved_default = self._resolve(default) if default is not None else None
        self._conditional_edges.append(
            ConditionalEdge(
                self._resolve(source),
                router,
                resolved_path_map,
                resolved_default,
            )
        )

    def add_parallel_edges(
        self,
        source: NodeLike[StateT],
        *targets: NodeLike[StateT],
        join: NodeTarget[StateT] = END,
    ) -> None:
        """Fan out from ``source`` to ``targets``, running them concurrently.

        Each target executes on an isolated copy of the current state. Results
        are merged back with the field reducers (unannotated fields are
        last-writer-wins), then execution continues at ``join`` — a node or
        :data:`~pyantra.graph.edge.END` (the default).

        Branches should return their updates explicitly; a branch that returns
        ``None`` contributes nothing.
        """
        if not targets:
            raise GraphCompileError(
                "add_parallel_edges() requires at least one target."
            )
        self._parallel_edges.append(
            ParallelEdge(
                source=self._resolve(source),
                targets=tuple(self._resolve(t) for t in targets),
                join=self._resolve_target(join),
            )
        )

    def compile(self) -> CompiledGraph[StateT]:
        """Validate the graph and return an executable ``CompiledGraph``."""
        from pyantra.graph.compiler import compile_graph

        return compile_graph(self)

    def _resolve(self, item: NodeLike[StateT]) -> str:
        if isinstance(item, Node):
            return item.name
        return item

    def _resolve_target(self, target: NodeTarget[StateT]) -> str | None:
        if isinstance(target, _End):
            return None
        return self._resolve(target)
