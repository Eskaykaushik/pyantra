"""Example: state reducers, parallel fan-out/join, and human-in-the-loop.

Run with:

    python examples/advanced_workflow.py
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated

from pyantra import (
    END,
    Graph,
    MemoryCheckpointStore,
    RunStatus,
    interrupt,
)


@dataclass
class State:
    query: str = ""
    answers: Annotated[list[str], operator.add] = field(default_factory=list)
    summary: str = ""
    verdict: str = ""


def parallel_search() -> None:
    graph = Graph(State)

    @graph.node
    def start(state: State) -> dict[str, str]:
        return {"query": "agent orchestration"}

    @graph.node
    def search_web(state: State) -> dict[str, list[str]]:
        time.sleep(0.05)
        return {"answers": ["web result"]}

    @graph.node
    def search_docs(state: State) -> dict[str, list[str]]:
        time.sleep(0.05)
        return {"answers": ["docs result"]}

    @graph.node
    def search_db(state: State) -> dict[str, list[str]]:
        time.sleep(0.05)
        return {"answers": ["db result"]}

    @graph.node
    def join(state: State) -> dict[str, str]:
        return {"summary": "; ".join(state.answers)}

    graph.set_entry_point(start)
    graph.add_parallel_edges(start, search_web, search_docs, search_db, join=join)

    result = graph.compile().run(State())
    assert result.status == RunStatus.COMPLETED
    assert len(result.state.answers) == 3
    print(f"parallel: summary={result.state.summary!r}")


def human_in_the_loop() -> None:
    store: MemoryCheckpointStore[State] = MemoryCheckpointStore()
    graph = Graph(State)

    @graph.node
    def review(state: State) -> dict[str, str]:
        decision = interrupt({"question": "approve?", "summary": state.summary})
        return {"verdict": decision}

    graph.set_entry_point(review)
    graph.add_edge(review, END)

    app = graph.compile()
    run = app.run(State(summary="a summary"), checkpointer=store, run_id="review-1")

    assert run.status == RunStatus.PAUSED
    print(f"interrupt: status={run.status.value}, payload={run.interrupt}")

    resumed = app.resume("review-1", "approved", checkpointer=store)
    assert resumed.status == RunStatus.COMPLETED
    print(f"resumed: verdict={resumed.state.verdict!r}")


def main() -> None:
    parallel_search()
    human_in_the_loop()
    print("All examples passed.")


if __name__ == "__main__":
    main()
