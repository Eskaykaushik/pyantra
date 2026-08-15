"""Tests for A2A types, the JSON-RPC client, and DelegateNode."""

from __future__ import annotations

import contextlib
import itertools
import json
import operator
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Annotated, Any

import pytest

from pyantra import (
    END,
    A2aClient,
    A2aError,
    AgentCard,
    DelegateNode,
    Graph,
    GraphCompileError,
    MemoryCheckpointStore,
    NodeExecutionError,
    RunStatus,
    Task,
    TaskStatus,
)
from pyantra.a2a import Message, TextPart
from pyantra.a2a.types import DataPart, FilePart

# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


def test_text_part_round_trip() -> None:
    part = TextPart(text="hello", metadata={"k": "v"})
    data = part.to_dict()
    assert data["kind"] == "text"
    assert TextPart.from_dict(data) == part


def test_data_and_file_parts_round_trip() -> None:
    data_part = DataPart(data={"n": 1})
    assert DataPart.from_dict(data_part.to_dict()) == data_part
    file_part = FilePart(file="https://example.com/a.pdf")
    assert FilePart.from_dict(file_part.to_dict()) == file_part


def test_message_round_trip() -> None:
    message = Message(
        role="user",
        parts=[TextPart(text="hi"), DataPart(data={"x": 1})],
        task_id="task-1",
    )
    restored = Message.from_dict(message.to_dict())
    assert restored == message
    assert restored.task_id == "task-1"


def test_task_round_trip_and_unknown_status() -> None:
    task = Task(
        id="t1",
        status=TaskStatus.COMPLETED,
        messages=[Message(role="agent", parts=[TextPart(text="done")])],
    )
    restored = Task.from_dict(task.to_dict())
    assert restored == task
    unknown = Task.from_dict({"id": "t2", "status": "weird"})
    assert unknown.status is TaskStatus.UNKNOWN


def test_agent_card_round_trip() -> None:
    card = AgentCard(
        name="assistant",
        description="helper",
        url="https://a.example/rpc",
        version="2.0",
        skills=[{"id": "s1", "name": "skill"}],
    )
    restored = AgentCard.from_dict(card.to_dict())
    assert restored == card


# --------------------------------------------------------------------------
# JSON-RPC client over a real local HTTP server
# --------------------------------------------------------------------------


class FakeA2A:
    """Stateful stand-in for a remote agent's A2A endpoint."""

    def __init__(
        self,
        card: dict[str, Any] | None = None,
        *,
        serve_well_known: bool = True,
    ) -> None:
        self.card = card or {
            "name": "fake",
            "description": "test agent",
            "version": "1.0",
        }
        self.serve_well_known = serve_well_known
        self.tasks: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.rpc_errors: dict[str, dict[str, Any]] = {}
        self._next = itertools.count(1)

    def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append({"method": method, "params": params})
        if method in self.rpc_errors:
            return self.rpc_errors[method]
        if method == "tasks/send":
            task_id = params.get("id") or f"task-{next(self._next)}"
            task = {
                "id": task_id,
                "status": "completed",
                "messages": [params["message"]],
            }
            self.tasks[task_id] = task
            return task
        if method == "tasks/get":
            return self.tasks.get(params["id"], {})
        if method == "message/send":
            task_id = params["id"]
            task = self.tasks.setdefault(
                task_id, {"id": task_id, "status": "completed", "messages": []}
            )
            text = params["message"]["parts"][0]["text"]
            task["messages"].append(
                {"role": "agent", "parts": [{"kind": "text", "text": f"pong {text}"}]}
            )
            return task
        return {"code": -32601, "message": f"Method not found: {method}"}


def _make_handler(fake: FakeA2A) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/.well-known/agent.json", "/agent.json"):
                if self.path.startswith("/.well-known") and not fake.serve_well_known:
                    self.send_error(404)
                else:
                    self._respond(json.dumps(fake.card))
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
            fake.requests.append(request)
            result = fake.handle(request.get("method", ""), request.get("params") or {})
            if "code" in result and "message" in result:
                body = json.dumps(
                    {"jsonrpc": "2.0", "id": request.get("id"), "error": result}
                )
            else:
                body = json.dumps(
                    {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
                )
            self._respond(body)

        def _respond(self, body: str) -> None:
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


@contextlib.contextmanager
def _a2a_server(fake: FakeA2A) -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(fake))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


async def test_client_fetch_agent_card() -> None:
    fake = FakeA2A()
    with _a2a_server(fake) as base:
        client = A2aClient()
        card = await client.fetch_agent_card(base)
        assert card.name == "fake"
        assert client.card is card


async def test_client_fetch_agent_card_falls_back_to_agent_json() -> None:
    fake = FakeA2A(card={"name": "fallback", "url": ""}, serve_well_known=False)
    with _a2a_server(fake) as base:
        client = A2aClient()
        card = await client.fetch_agent_card(base)
        assert card.name == "fallback"


async def test_client_send_get_message_round_trip() -> None:
    fake = FakeA2A()
    with _a2a_server(fake) as base:
        client = A2aClient(agent_url=base)
        task = await client.send_task(Message(role="user", parts=[TextPart(text="hi")]))
        assert task.id == "task-1"
        assert task.status is TaskStatus.COMPLETED

        got = await client.get_task("task-1")
        assert got.id == "task-1"

        replied = await client.send_message(
            "task-1", Message(role="user", parts=[TextPart(text="more")])
        )
        assert replied.status is TaskStatus.COMPLETED
        texts = [
            part.text
            for m in replied.messages
            for part in m.parts
            if isinstance(part, TextPart)
        ]
        assert "pong more" in texts


async def test_client_rpc_error_raises_a2a_error() -> None:
    fake = FakeA2A()
    fake.rpc_errors["tasks/get"] = {"code": -32000, "message": "boom"}
    with _a2a_server(fake) as base:
        client = A2aClient(agent_url=base)
        with pytest.raises(A2aError, match="boom"):
            await client.get_task("task-1")


async def test_client_unknown_method_raises() -> None:
    fake = FakeA2A()
    with _a2a_server(fake) as base:
        client = A2aClient(agent_url=base)
        with pytest.raises(A2aError, match="Method not found"):
            await client.cancel_task("task-1")


async def test_client_requires_url_before_rpc() -> None:
    with pytest.raises(A2aError, match="agent URL"):
        await A2aClient().get_task("task-1")


# --------------------------------------------------------------------------
# DelegateNode
# --------------------------------------------------------------------------


@dataclass
class A2aState:
    prompt: str = ""
    result: str = ""
    task_id: str = ""
    history: Annotated[list[str], operator.add] = field(default_factory=list)


class FakeA2AClient:
    """Duck-typed stand-in for A2aClient with a scripted task lifecycle."""

    def __init__(
        self,
        *,
        send_status: str = "completed",
        gets: list[str] = (),
        result: str = "agent reply",
    ) -> None:
        self._send_status = TaskStatus(send_status)
        self.gets = list(gets)
        self.result = result
        self.calls: list[tuple[Any, ...]] = []
        self.sent_inputs: list[Message] = []
        self._counter = itertools.count(1)

    async def send_task(
        self,
        message: Message,
        *,
        task_id: str | None = None,
        metadata: dict | None = None,
    ) -> Task:
        self.calls.append(("send_task", message.to_dict(), task_id))
        task = Task(
            id=task_id or f"task-{next(self._counter)}",
            status=self._send_status,
            messages=[message],
        )
        return self._resolve(task)

    async def get_task(self, task_id: str) -> Task:
        self.calls.append(("get_task", task_id))
        status = TaskStatus(self.gets.pop(0)) if self.gets else TaskStatus.COMPLETED
        return self._resolve(Task(id=task_id, status=status))

    async def send_message(self, task_id: str, message: Message) -> Task:
        self.calls.append(("send_message", task_id, message.to_dict()))
        self.sent_inputs.append(message)
        return self._resolve(Task(id=task_id, status=TaskStatus.COMPLETED))

    def _resolve(self, task: Task) -> Task:
        if task.status is TaskStatus.COMPLETED:
            task.messages = [
                Message(role="agent", parts=[TextPart(text=self.result)])
            ]
        elif task.status is TaskStatus.INPUT_REQUIRED:
            task.messages = [
                Message(role="agent", parts=[TextPart(text="please clarify")])
            ]
        return task


def _delegate(
    client: FakeA2AClient,
    *,
    result_field: str = "result",
    task_id_field: str = "task_id",
    **kwargs: Any,
) -> Graph:
    graph = Graph(A2aState)
    node = DelegateNode(
        name="delegate",
        client=client,
        result_field=result_field,
        payload_from=lambda state: Message(
            role="user", parts=[TextPart(text=state.prompt)]
        ),
        task_id_field=task_id_field,
        **kwargs,
    )
    graph.add_node(node)
    graph.set_entry_point(node)
    graph.add_edge(node, END)
    return graph


def test_delegate_node_completes_and_merges() -> None:
    client = FakeA2AClient()
    app = _delegate(client).compile()
    run = app.run(A2aState(prompt="do it"))

    assert run.status is RunStatus.COMPLETED
    sent = Message(role="user", parts=[TextPart(text="do it")])
    assert client.calls == [("send_task", sent.to_dict(), None)]
    assert run.state.result == "agent reply"
    assert run.state.task_id == "task-1"


def test_delegate_node_polls_until_terminal() -> None:
    client = FakeA2AClient(send_status="working", gets=["working", "completed"])
    run = _delegate(client).compile().run(A2aState())

    assert run.status is RunStatus.COMPLETED
    assert [name for name, *_ in client.calls] == ["send_task", "get_task", "get_task"]


def test_delegate_node_reuses_existing_task_id() -> None:
    client = FakeA2AClient(send_status="working", gets=["completed"])
    run = _delegate(client).compile().run(A2aState(task_id="existing-1"))

    assert run.status is RunStatus.COMPLETED
    assert client.calls == [("get_task", "existing-1")]
    assert run.state.result == "agent reply"


def test_delegate_node_failed_task_fails_run() -> None:
    client = FakeA2AClient(send_status="failed")
    run = _delegate(client).compile().run(A2aState())

    assert run.status is RunStatus.FAILED
    assert isinstance(run.exception, NodeExecutionError)
    assert isinstance(run.exception.__cause__, A2aError)
    assert "failed" in str(run.exception.__cause__)


def test_delegate_node_unknown_task_fails_run() -> None:
    client = FakeA2AClient(send_status="unknown")
    run = _delegate(client).compile().run(A2aState())
    assert run.status is RunStatus.FAILED
    assert isinstance(run.exception.__cause__, A2aError)
    assert "unknown" in str(run.exception.__cause__)


def test_delegate_node_max_wait_timeout() -> None:
    class StuckClient(FakeA2AClient):
        async def get_task(self, task_id: str) -> Task:
            self.calls.append(("get_task", task_id))
            return Task(id=task_id, status=TaskStatus.WORKING)

    run = _delegate(
        StuckClient(send_status="working"),
        poll_interval=0.01,
        max_wait=0.1,
    ).compile().run(A2aState())

    assert run.status is RunStatus.FAILED
    assert isinstance(run.exception.__cause__, A2aError)
    assert "max_wait" in str(run.exception.__cause__)


def test_delegate_node_input_required_pauses_and_resumes() -> None:
    store: MemoryCheckpointStore[A2aState] = MemoryCheckpointStore()
    client = FakeA2AClient(send_status="input-required")
    app = _delegate(client).compile()

    run = app.run(A2aState(prompt="go"), checkpointer=store, run_id="delegate-1")

    assert run.status is RunStatus.PAUSED
    assert run.interrupt["kind"] == "a2a.input-required"
    assert run.interrupt["task"]["id"] == "task-1"
    assert run.interrupt["message"]["role"] == "agent"

    resumed = app.resume("delegate-1", "the answer", checkpointer=store)

    assert resumed.status is RunStatus.COMPLETED
    assert [m.parts[0].text for m in client.sent_inputs] == ["the answer"]
    assert resumed.state.result == "agent reply"


def test_delegate_node_input_required_resume_without_task_id_field() -> None:
    store: MemoryCheckpointStore[A2aState] = MemoryCheckpointStore()
    client = FakeA2AClient(send_status="input-required")
    app = _delegate(client, task_id_field=None).compile()

    run = app.run(A2aState(prompt="go"), checkpointer=store, run_id="delegate-nofield")
    assert run.status is RunStatus.PAUSED
    assert run.interrupt["task"]["id"] == "task-1"

    resumed = app.resume("delegate-nofield", "the answer", checkpointer=store)
    assert resumed.status is RunStatus.COMPLETED
    assert [m.parts[0].text for m in client.sent_inputs] == ["the answer"]
    assert resumed.state.result == "agent reply"


def test_delegate_node_input_required_via_polling() -> None:
    store: MemoryCheckpointStore[A2aState] = MemoryCheckpointStore()
    client = FakeA2AClient(send_status="working", gets=["input-required"])
    app = _delegate(client).compile()

    run = app.run(A2aState(), checkpointer=store, run_id="delegate-2")
    assert run.status is RunStatus.PAUSED
    assert run.interrupt["task"]["id"] == "task-1"

    resumed = app.resume("delegate-2", "ok", checkpointer=store)
    assert resumed.status is RunStatus.COMPLETED
    assert [m.parts[0].text for m in client.sent_inputs] == ["ok"]


def test_delegate_node_input_required_requires_checkpointer() -> None:
    client = FakeA2AClient(send_status="input-required")
    run = _delegate(client).compile().run(A2aState())

    assert run.status is RunStatus.FAILED
    assert "checkpointer" in str(run.exception)


def test_delegate_node_multi_turn_negotiation() -> None:
    store: MemoryCheckpointStore[A2aState] = MemoryCheckpointStore()

    class MultiTurnClient(FakeA2AClient):
        def __init__(self) -> None:
            super().__init__(send_status="input-required")
            self.rounds = 0

        async def send_message(self, task_id: str, message: Message) -> Task:
            self.sent_inputs.append(message)
            self.rounds += 1
            if self.rounds == 1:
                return self._resolve(Task(id=task_id, status=TaskStatus.INPUT_REQUIRED))
            return self._resolve(Task(id=task_id, status=TaskStatus.COMPLETED))

    client = MultiTurnClient()
    app = _delegate(client).compile()

    run = app.run(A2aState(), checkpointer=store, run_id="delegate-3")
    assert run.status is RunStatus.PAUSED

    run = app.resume("delegate-3", "first", checkpointer=store)
    assert run.status is RunStatus.PAUSED

    run = app.resume("delegate-3", "second", checkpointer=store)
    assert run.status is RunStatus.COMPLETED
    assert [m.parts[0].text for m in client.sent_inputs] == ["first", "second"]


def test_delegate_node_custom_result_from() -> None:
    client = FakeA2AClient()
    graph = Graph(A2aState)
    node = DelegateNode(
        name="delegate",
        client=client,
        result_field="result",
        result_from=lambda state, task: task.id,
        payload_from=lambda state: Message(role="user", parts=[TextPart(text="x")]),
    )
    graph.add_node(node)
    graph.set_entry_point(node)
    graph.add_edge(node, END)

    run = graph.compile().run(A2aState())
    assert run.state.result == "task-1"


def test_delegate_node_composes_with_preceding_node() -> None:
    client = FakeA2AClient()
    graph = Graph(A2aState)
    node = DelegateNode(
        name="delegate",
        client=client,
        result_field="result",
        payload_from=lambda state: Message(
            role="user", parts=[TextPart(text=state.prompt)]
        ),
    )
    graph.add_node(node)

    @graph.node
    def prepare(state: A2aState) -> dict[str, str]:
        return {"prompt": "prepared"}

    graph.set_entry_point(prepare)
    graph.add_edge(prepare, node)
    graph.add_edge(node, END)

    run = graph.compile().run(A2aState())
    assert run.status is RunStatus.COMPLETED
    assert run.state.result == "agent reply"


def test_delegate_node_validates_fields() -> None:
    graph = Graph(A2aState)
    bad = DelegateNode(
        name="delegate", client=FakeA2AClient(), result_field="bogus"
    )
    graph.add_node(bad)
    graph.set_entry_point(bad)
    graph.add_edge(bad, END)
    with pytest.raises(GraphCompileError, match="bogus"):
        graph.compile()

    graph = Graph(A2aState)
    bad_task = DelegateNode(
        name="delegate",
        client=FakeA2AClient(),
        result_field="result",
        task_id_field="nope",
    )
    graph.add_node(bad_task)
    graph.set_entry_point(bad_task)
    graph.add_edge(bad_task, END)
    with pytest.raises(GraphCompileError, match="nope"):
        graph.compile()


def test_a2a_imports_do_not_pull_in_deps() -> None:
    code = (
        "import sys\n"
        "import pyantra.a2a\n"
        "for name in ('sqlalchemy', 'dbos', 'httpx', 'aiohttp'):\n"
        "    assert name not in sys.modules, f'{name} imported by pyantra.a2a'\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
