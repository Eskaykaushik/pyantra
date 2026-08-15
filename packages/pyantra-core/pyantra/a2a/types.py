"""A2A (Agent-to-Agent) protocol types.

Stdlib-only dataclasses mirroring the A2A protocol's core objects — the
``AgentCard``, ``Task`` lifecycle, ``Message``, and content parts — with
``to_dict``/``from_dict`` round-trips so they map directly onto the JSON
payloads exchanged over JSON-RPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from pyantra.a2a.errors import A2aError


class TaskStatus(str, Enum):
    """Lifecycle of an A2A task (mirrors ``TaskStatus`` in the spec)."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class TextPart:
    """Plain-text content part."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: ClassVar[str] = "text"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "text": self.text}
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextPart:
        return cls(text=str(data["text"]), metadata=data.get("metadata") or {})


@dataclass
class DataPart:
    """Structured data content part."""

    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: ClassVar[str] = "data"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "data": self.data}
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataPart:
        return cls(data=data.get("data"), metadata=data.get("metadata") or {})


@dataclass
class FilePart:
    """File content part, referenced by URI."""

    file: str
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: ClassVar[str] = "file"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "file": self.file}
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilePart:
        return cls(file=str(data["file"]), metadata=data.get("metadata") or {})


Part = TextPart | DataPart | FilePart


def part_from_dict(data: dict[str, Any]) -> Part:
    """Rebuild a content part from its wire representation."""
    kind = data.get("kind")
    if kind == "text":
        return TextPart.from_dict(data)
    if kind == "data":
        return DataPart.from_dict(data)
    if kind == "file":
        return FilePart.from_dict(data)
    raise A2aError(f"Unknown A2A part kind {kind!r}.")


@dataclass
class Message:
    """A message in an A2A task exchange (role ``user`` or ``agent``)."""

    role: str
    parts: list[Part]
    task_id: str | None = None
    context_id: str | None = None
    message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.task_id is not None:
            data["taskId"] = self.task_id
        if self.context_id is not None:
            data["contextId"] = self.context_id
        if self.message_id is not None:
            data["messageId"] = self.message_id
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=str(data["role"]),
            parts=[part_from_dict(part) for part in data.get("parts") or []],
            task_id=data.get("taskId"),
            context_id=data.get("contextId"),
            message_id=data.get("messageId"),
            metadata=data.get("metadata") or {},
        )


@dataclass
class Task:
    """A unit of work delegated to an agent.

    ``status`` moves through ``submitted``/``working`` to a terminal state;
    ``input-required`` pauses for human (or orchestrator) input via the
    ``message/send`` negotiation method.
    """

    id: str
    status: TaskStatus
    messages: list[Message] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status.value,
            "messages": [message.to_dict() for message in self.messages],
            "artifacts": self.artifacts,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        try:
            status = TaskStatus(str(data["status"]))
        except (KeyError, ValueError):
            status = TaskStatus.UNKNOWN
        return cls(
            id=str(data.get("id") or data.get("taskId") or ""),
            status=status,
            messages=[Message.from_dict(item) for item in data.get("messages") or []],
            artifacts=[dict(item) for item in data.get("artifacts") or []],
            metadata=data.get("metadata") or {},
        )


@dataclass
class AgentCard:
    """Metadata describing a remote agent, served at ``/.well-known/agent.json``."""

    name: str
    description: str = ""
    url: str = ""
    version: str = ""
    skills: list[dict[str, Any]] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name}
        if self.description:
            data["description"] = self.description
        if self.url:
            data["url"] = self.url
        if self.version:
            data["version"] = self.version
        if self.skills:
            data["skills"] = self.skills
        if self.capabilities:
            data["capabilities"] = self.capabilities
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCard:
        return cls(
            name=str(data.get("name") or ""),
            description=data.get("description") or "",
            url=data.get("url") or "",
            version=data.get("version") or "",
            skills=[dict(skill) for skill in data.get("skills") or []],
            capabilities=[str(item) for item in data.get("capabilities") or []],
        )


__all__ = [
    "AgentCard",
    "DataPart",
    "FilePart",
    "Message",
    "Part",
    "Task",
    "TaskStatus",
    "TextPart",
    "part_from_dict",
]
