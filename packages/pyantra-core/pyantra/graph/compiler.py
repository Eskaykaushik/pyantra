"""Graph compiler: validation and the compiled, executable graph."""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Generic

from pyantra.checkpoint.base import CheckpointStore
from pyantra.graph.conditional import ConditionalEdge
from pyantra.graph.edge import Edge
from pyantra.graph.node import Node
from pyantra.graph.parallel import ParallelEdge
from pyantra.runtime.errors import CheckpointError, GraphCompileError
from pyantra.runtime.run import Run
from pyantra.state.reducers import Reducer
from pyantra.state.state import StateT

if TYPE_CHECKING:
    from pyantra.graph.graph import Graph

_MAX_ITERATIONS_DEFAULT = 100


class CompiledGraph(Generic[StateT]):
    """A validated graph ready to be executed.

    Returned by :meth:`Graph.compile`. Call ``run`` (sync) or ``arun`` (async)
    to execute it.
    """

    def __init__(
        self,
        state_type: type[StateT],
        entry_point: str,
        nodes: Mapping[str, Node[StateT]],
        edges: Mapping[str, Sequence[Edge]],
        conditional_edges: Mapping[str, Sequence[ConditionalEdge[StateT]]],
        parallel_edges: Mapping[str, Sequence[ParallelEdge]] | None = None,
        reducers: Mapping[str, Reducer] | None = None,
    ) -> None:
        self._state_type = state_type
        self._entry_point = entry_point
        self._nodes: dict[str, Node[StateT]] = dict(nodes)
        self._edges: dict[str, list[Edge]] = {
            source: list(edges) for source, edges in edges.items()
        }
        self._conditional_edges: dict[str, list[ConditionalEdge[StateT]]] = {
            source: list(edges) for source, edges in conditional_edges.items()
        }
        self._parallel_edges: dict[str, list[ParallelEdge]] = {
            source: list(edges) for source, edges in (parallel_edges or {}).items()
        }
        self._reducers: dict[str, Reducer] = dict(reducers or {})

    @property
    def state_type(self) -> type[StateT]:
        return self._state_type

    @property
    def reducers(self) -> Mapping[str, Reducer]:
        """Field reducers extracted from the state type."""
        return dict(self._reducers)

    @property
    def parallel_edges(self) -> Mapping[str, Sequence[ParallelEdge]]:
        return {source: list(edges) for source, edges in self._parallel_edges.items()}

    @property
    def entry_point(self) -> str:
        return self._entry_point

    @property
    def nodes(self) -> Mapping[str, Node[StateT]]:
        return dict(self._nodes)

    @property
    def edges(self) -> Mapping[str, Sequence[Edge]]:
        return {source: list(edges) for source, edges in self._edges.items()}

    @property
    def conditional_edges(self) -> Mapping[str, Sequence[ConditionalEdge[StateT]]]:
        return {
            source: list(edges) for source, edges in self._conditional_edges.items()
        }

    def run(
        self,
        state: StateT,
        *,
        max_iterations: int = _MAX_ITERATIONS_DEFAULT,
        checkpointer: CheckpointStore[StateT] | None = None,
        run_id: str | None = None,
    ) -> Run[StateT]:
        """Execute the graph synchronously and return a ``Run``.

        ``checkpointer`` enables durable checkpoints so a failed run can be
        resumed by re-invoking ``run`` with the same ``run_id``, and an
        interrupted run can be resumed with :meth:`resume`.

        Raises ``RuntimeError`` when called from inside a running event loop;
        use :meth:`run_sync` there.
        """
        coro = self.arun(
            state,
            max_iterations=max_iterations,
            checkpointer=checkpointer,
            run_id=run_id,
        )
        try:
            return asyncio.run(coro)
        except RuntimeError:
            coro.close()
            raise

    def run_sync(
        self,
        state: StateT,
        *,
        max_iterations: int = _MAX_ITERATIONS_DEFAULT,
        checkpointer: CheckpointStore[StateT] | None = None,
        run_id: str | None = None,
    ) -> Run[StateT]:
        """Execute the graph synchronously even inside a running event loop.

        With no event loop running this behaves like :meth:`run`. When a loop
        is already running (e.g. inside a FastAPI handler or an async node),
        the graph runs to completion on a fresh event loop in a worker thread
        and this method blocks until it finishes.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    state,
                    max_iterations=max_iterations,
                    checkpointer=checkpointer,
                    run_id=run_id,
                )
            )

        result: Run[StateT] | None = None
        error: BaseException | None = None

        def target() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(
                    self.arun(
                        state,
                        max_iterations=max_iterations,
                        checkpointer=checkpointer,
                        run_id=run_id,
                    )
                )
            except BaseException as exc:
                error = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        assert result is not None
        return result

    async def arun(
        self,
        state: StateT,
        *,
        max_iterations: int = _MAX_ITERATIONS_DEFAULT,
        checkpointer: CheckpointStore[StateT] | None = None,
        run_id: str | None = None,
        interrupt_responses: dict[str, object] | None = None,
    ) -> Run[StateT]:
        """Execute the graph asynchronously and return a ``Run``."""
        from pyantra.runtime.executor import Executor

        return await Executor(self).arun(
            state,
            max_iterations=max_iterations,
            checkpointer=checkpointer,
            run_id=run_id,
            interrupt_responses=interrupt_responses,
        )

    def resume(
        self,
        run_id: str,
        interrupt_value: object,
        *,
        checkpointer: CheckpointStore[StateT],
        max_iterations: int = _MAX_ITERATIONS_DEFAULT,
    ) -> Run[StateT]:
        """Resume a paused run with the value requested by its interrupt.

        Raises ``RuntimeError`` when called from inside a running event loop.
        """
        coro = self.aresume(
            run_id,
            interrupt_value,
            checkpointer=checkpointer,
            max_iterations=max_iterations,
        )
        try:
            return asyncio.run(coro)
        except RuntimeError:
            coro.close()
            raise

    def resume_sync(
        self,
        run_id: str,
        interrupt_value: object,
        *,
        checkpointer: CheckpointStore[StateT],
        max_iterations: int = _MAX_ITERATIONS_DEFAULT,
    ) -> Run[StateT]:
        """Resume a paused run synchronously even inside a running event loop.

        With no event loop running this behaves like :meth:`resume`. When a
        loop is already running (e.g. inside a FastAPI handler or an async
        node), the resumed graph runs to completion on a fresh event loop in
        a worker thread and this method blocks until it finishes.
        """
        coro = self.aresume(
            run_id,
            interrupt_value,
            checkpointer=checkpointer,
            max_iterations=max_iterations,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Run[StateT] | None = None
        error: BaseException | None = None

        def target() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:
                error = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        assert result is not None
        return result

    async def aresume(
        self,
        run_id: str,
        interrupt_value: object,
        *,
        checkpointer: CheckpointStore[StateT],
        max_iterations: int = _MAX_ITERATIONS_DEFAULT,
    ) -> Run[StateT]:
        """Resume a paused run asynchronously with the interrupt response."""
        checkpoint = checkpointer.load(run_id)
        if checkpoint is None:
            raise CheckpointError(
                f"No checkpoint found for run {run_id!r}; cannot resume."
            )
        node = checkpoint.resume_at
        if checkpoint.interrupts:
            node = checkpoint.interrupts[-1][0]
        if node is None:
            raise CheckpointError(f"Run {run_id!r} has no node to resume from.")
        return await self.arun(
            checkpoint.state,
            max_iterations=max_iterations,
            checkpointer=checkpointer,
            run_id=run_id,
            interrupt_responses={node: interrupt_value},
        )


def compile_graph(graph: Graph[StateT]) -> CompiledGraph[StateT]:
    """Validate and compile a :class:`~pyantra.graph.graph.Graph`.

    Validation is intentionally strict: the graph fails early at compile time
    rather than at runtime.
    """
    validate(graph)

    edges: dict[str, list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        edges[edge.source].append(edge)

    conditional_edges: dict[str, list[ConditionalEdge[StateT]]] = defaultdict(list)
    for cond_edge in graph.conditional_edges:
        conditional_edges[cond_edge.source].append(cond_edge)

    parallel_edges: dict[str, list[ParallelEdge]] = defaultdict(list)
    for par_edge in graph.parallel_edges:
        parallel_edges[par_edge.source].append(par_edge)

    assert graph.entry_point is not None
    return CompiledGraph(
        state_type=graph.state_type,
        entry_point=graph.entry_point,
        nodes=graph.nodes,
        edges=edges,
        conditional_edges=conditional_edges,
        parallel_edges=parallel_edges,
        reducers=graph.reducers,
    )


def validate(graph: Graph[StateT]) -> None:
    """Validate a graph definition, raising ``GraphCompileError`` on failure."""
    nodes = graph.nodes
    entry = graph.entry_point

    if entry is None:
        raise GraphCompileError(
            "Graph has no entry point. Call set_entry_point() before compile()."
        )
    if entry not in nodes:
        raise GraphCompileError(f"Entry point {entry!r} is not a registered node.")

    for node in nodes.values():
        node.validate(graph.state_type)

    for edge in graph.edges:
        if edge.source not in nodes:
            raise GraphCompileError(
                f"Edge source {edge.source!r} is not a registered node."
            )
        if edge.target is not None and edge.target not in nodes:
            raise GraphCompileError(
                f"Edge from {edge.source!r} references unknown node {edge.target!r}."
            )

    conditional_sources: set[str] = set()
    for cond_edge in graph.conditional_edges:
        if cond_edge.source not in nodes:
            raise GraphCompileError(
                f"Conditional edge source {cond_edge.source!r} is not a "
                "registered node."
            )
        if cond_edge.source in conditional_sources:
            raise GraphCompileError(
                f"Node {cond_edge.source!r} has more than one conditional edge."
            )
        conditional_sources.add(cond_edge.source)
        if cond_edge.path_map is not None:
            for key, target in cond_edge.path_map.items():
                if target not in nodes:
                    raise GraphCompileError(
                        f"Conditional route {key!r} from {cond_edge.source!r} "
                        f"references unknown node {target!r}."
                    )
        if cond_edge.default is not None:
            if cond_edge.path_map is None:
                raise GraphCompileError(
                    f"A default route requires a path_map on the conditional edge "
                    f"from {cond_edge.source!r}."
                )
            if cond_edge.default not in nodes:
                raise GraphCompileError(
                    f"Default route {cond_edge.default!r} from {cond_edge.source!r} "
                    f"references an unknown node."
                )

    normal_sources = {edge.source for edge in graph.edges}
    parallel_sources = {par_edge.source for par_edge in graph.parallel_edges}
    overlap = normal_sources & conditional_sources
    if overlap:
        raise GraphCompileError(
            f"Node(s) {sorted(overlap)} define both normal and conditional outgoing "
            "edges; use one or the other."
        )
    parallel_overlap = (normal_sources | conditional_sources) & parallel_sources
    if parallel_overlap:
        raise GraphCompileError(
            f"Node(s) {sorted(parallel_overlap)} define both a parallel fan-out and "
            "other outgoing edges; use one or the other."
        )

    for par_edge in graph.parallel_edges:
        if par_edge.source not in nodes:
            raise GraphCompileError(
                f"Parallel edge source {par_edge.source!r} is not a registered node."
            )
        for target in par_edge.targets:
            if target not in nodes:
                raise GraphCompileError(
                    f"Parallel fan-out from {par_edge.source!r} references "
                    f"unknown node {target!r}."
                )
        if par_edge.join is not None and par_edge.join not in nodes:
            raise GraphCompileError(
                f"Parallel join from {par_edge.source!r} references unknown "
                f"node {par_edge.join!r}."
            )

    from collections import Counter

    out_degree = Counter(edge.source for edge in graph.edges)
    ambiguous = {source for source, count in out_degree.items() if count > 1}
    if ambiguous:
        raise GraphCompileError(
            f"Node(s) {sorted(ambiguous)} have multiple unconditional outgoing "
            "edges; use add_conditional_edges() for branching."
        )

    parallel_degree = Counter(par_edge.source for par_edge in graph.parallel_edges)
    duplicated_parallel = {
        source for source, count in parallel_degree.items() if count > 1
    }
    if duplicated_parallel:
        raise GraphCompileError(
            f"Node(s) {sorted(duplicated_parallel)} have more than one parallel "
            "fan-out; use a single add_parallel_edges() call with all targets."
        )

    reachable = _reachable(
        nodes, graph.edges, graph.conditional_edges, graph.parallel_edges, entry
    )
    unreachable = set(nodes) - reachable
    if unreachable:
        raise GraphCompileError(
            f"Nodes unreachable from entry point {entry!r}: "
            f"{', '.join(sorted(unreachable))}."
        )

    _detect_unconditional_cycles(nodes, graph.edges)


def _reachable(
    nodes: Mapping[str, Node[StateT]],
    edges: Sequence[Edge],
    conditional_edges: Sequence[ConditionalEdge[StateT]],
    parallel_edges: Sequence[ParallelEdge],
    entry: str,
) -> set[str]:
    """Compute nodes reachable from the entry point.

    A router-only conditional edge (no ``path_map``) may return any registered
    node, so such sources conservatively mark every node as reachable to avoid
    false positives.
    """
    outgoing: dict[str, list[str | None]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source].append(edge.target)
    for cond_edge in conditional_edges:
        if cond_edge.path_map is not None:
            outgoing[cond_edge.source].extend(cond_edge.path_map.values())
            if cond_edge.default is not None:
                outgoing[cond_edge.source].append(cond_edge.default)
        else:
            outgoing[cond_edge.source].extend(nodes)
    for par_edge in parallel_edges:
        outgoing[par_edge.source].extend(par_edge.targets)
        if par_edge.join is not None:
            outgoing[par_edge.source].append(par_edge.join)

    seen: set[str] = set()
    queue: deque[str] = deque([entry])
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for target in outgoing.get(node, []):
            if target is not None and target not in seen:
                queue.append(target)
    return seen


def _detect_unconditional_cycles(
    nodes: Mapping[str, Node[StateT]],
    edges: Sequence[Edge],
) -> None:
    """Detect cycles composed only of unconditional edges.

    Such cycles have no conditional exit and can never terminate.
    """
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.target is not None:
            outgoing[edge.source].append(edge.target)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        color[node] = GRAY
        path.append(node)
        for nxt in outgoing[node]:
            if color.get(nxt, WHITE) == GRAY:
                cycle = path[path.index(nxt) :] + [nxt]
                raise GraphCompileError(
                    "Graph contains an unconditional cycle with no exit: "
                    f"{' -> '.join(cycle)}."
                )
            if color.get(nxt, WHITE) == WHITE:
                visit(nxt, path)
        path.pop()
        color[node] = BLACK

    for node in nodes:
        if color.get(node, WHITE) == WHITE:
            visit(node, [])
