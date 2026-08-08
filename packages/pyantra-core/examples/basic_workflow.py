"""End-to-end example: a linear workflow and a conditional workflow.

Run with:

    python examples/basic_workflow.py
"""

from __future__ import annotations

from dataclasses import dataclass

from pyantra import Graph


@dataclass
class State:
    value: int


def linear_workflow() -> int:
    graph = Graph(State)

    @graph.node
    def increment(state: State) -> State:
        state.value += 1
        return state

    @graph.node
    def double(state: State) -> State:
        state.value *= 2
        return state

    graph.set_entry_point(increment)
    graph.add_edge(increment, double)

    app = graph.compile()
    result = app.run(State(value=1))

    assert result.state is not None
    print(f"linear: status={result.status.value}, value={result.state.value}")
    return result.state.value


def conditional_workflow() -> str:
    graph = Graph(State)

    @graph.node
    def classify(state: State) -> State:
        return state

    @graph.node
    def process_positive(state: State) -> State:
        state.value = abs(state.value)
        return state

    @graph.node
    def process_negative(state: State) -> State:
        state.value = -abs(state.value)
        return state

    def route(state: State) -> str:
        return "positive" if state.value >= 0 else "negative"

    graph.set_entry_point(classify)
    graph.add_conditional_edges(
        classify,
        route,
        {"positive": process_positive, "negative": process_negative},
    )

    app = graph.compile()

    positive = app.run(State(value=5))
    negative = app.run(State(value=-5))

    assert positive.state is not None and negative.state is not None
    print(f"positive: {positive.state.value}, negative: {negative.state.value}")
    return f"{positive.state.value}/{negative.state.value}"


async def async_workflow() -> int:

    graph = Graph(State)

    @graph.node
    async def increment(state: State) -> State:
        state.value += 1
        return state

    graph.set_entry_point(increment)

    result = await graph.compile().arun(State(value=41))

    assert result.state is not None
    print(f"async: value={result.state.value}")
    return result.state.value


def main() -> None:
    import asyncio

    assert linear_workflow() == 4
    assert conditional_workflow() == "5/-5"
    assert asyncio.run(async_workflow()) == 42
    print("All examples passed.")


if __name__ == "__main__":
    main()
