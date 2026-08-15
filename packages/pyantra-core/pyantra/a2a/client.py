"""Minimal A2A client over JSON-RPC 2.0.

The A2A protocol is plain HTTP + JSON-RPC, so the client needs nothing beyond
the standard library — pyantra core stays dependency-free. Calls are made on a
worker thread (``asyncio.to_thread``) so they compose with the async executor.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import urllib.error
import urllib.request
from typing import Any, Protocol, cast

from pyantra.a2a.errors import A2aError
from pyantra.a2a.types import AgentCard, Message, Task


class A2aClientProtocol(Protocol):
    """The interface :class:`~pyantra.a2a.DelegateNode` relies on.

    Duck-typed so tests and alternate transports can stand in for
    :class:`A2aClient`.
    """

    async def send_task(
        self,
        message: Message,
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task: ...

    async def get_task(self, task_id: str) -> Task: ...

    async def send_message(self, task_id: str, message: Message) -> Task: ...


class A2aClient:
    """A client for one remote agent, speaking A2A over JSON-RPC.

    ``agent_url`` is the agent's endpoint that accepts ``tasks/send``;
    ``agent_card`` may be supplied (or fetched with
    :meth:`fetch_agent_card`) so the agent's metadata is available without a
    separate round trip. If ``agent_url`` is omitted, the card's ``url`` is
    used. A client with neither is valid as long as an AgentCard is fetched
    before the first RPC call.

    Example::

        client = A2aClient(agent_url="https://agent.example.com/rpc")
        task = await client.send_task(
            Message(role="user", parts=[TextPart(text="translate this")])
        )
    """

    def __init__(
        self,
        *,
        agent_url: str | None = None,
        agent_card: AgentCard | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._url = agent_url or (agent_card.url if agent_card else None)
        self._card = agent_card
        self._timeout = timeout
        self._ids = itertools.count(1)

    @property
    def agent_url(self) -> str | None:
        """The agent's JSON-RPC endpoint, or None if not yet known."""
        return self._url

    @property
    def card(self) -> AgentCard | None:
        """The agent's metadata, if a card was supplied or fetched."""
        return self._card

    async def fetch_agent_card(self, base_url: str) -> AgentCard:
        """Fetch the agent's ``AgentCard`` from ``base_url``.

        Tries ``/.well-known/agent.json`` (the A2A convention), falling back
        to ``/agent.json``. The card's ``url`` becomes the RPC endpoint when
        none was configured.
        """
        base = base_url.rstrip("/")
        last_error: Exception | None = None
        for path in ("/.well-known/agent.json", "/agent.json"):
            try:
                data = await asyncio.to_thread(self._get_json, base + path)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 404:
                    raise A2aError(
                        f"Failed to fetch AgentCard from {base}{path}: "
                        f"HTTP {exc.code}."
                    ) from exc
                continue
            card = AgentCard.from_dict(data)
            self._card = card
            self._url = card.url or self._url
            return card
        raise A2aError(f"No AgentCard found at {base_url}.") from last_error

    async def send_task(
        self,
        message: Message,
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Send a task to the agent and return the resulting ``Task``."""
        params: dict[str, Any] = {"message": message.to_dict()}
        if task_id is not None:
            params["id"] = task_id
        if metadata:
            params["metadata"] = metadata
        return Task.from_dict(await self._rpc("tasks/send", params))

    async def get_task(self, task_id: str) -> Task:
        """Fetch the current state of a task by its id."""
        return Task.from_dict(await self._rpc("tasks/get", {"id": task_id}))

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task and return its (``canceled``) state."""
        return Task.from_dict(await self._rpc("tasks/cancel", {"id": task_id}))

    async def send_message(self, task_id: str, message: Message) -> Task:
        """Send a message to an in-progress task (e.g. its requested input)."""
        params: dict[str, Any] = {"id": task_id, "message": message.to_dict()}
        return Task.from_dict(await self._rpc("message/send", params))

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        if self._url is None:
            raise A2aError(
                "No agent URL configured; set agent_url or fetch an AgentCard."
            )
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": method,
                "params": params,
            }
        ).encode("utf-8")
        return await asyncio.to_thread(self._post, payload)

    def _post(self, payload: bytes) -> Any:
        assert self._url is not None
        request = urllib.request.Request(
            self._url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise A2aError(
                f"A2A request to {self._url} failed with HTTP {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')}"
            ) from exc
        if not isinstance(body, dict) or "result" not in body:
            error = body.get("error") if isinstance(body, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            message = (
                error.get("message") if isinstance(error, dict) else str(error)
            )
            raise A2aError(f"A2A RPC error {code}: {message}")
        return body["result"]

    def _get_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url, method="GET", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


__all__ = ["A2aClient", "A2aClientProtocol"]
