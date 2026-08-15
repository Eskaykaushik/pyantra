"""A2A (Agent-to-Agent) delegation for pyantra.

A2A is an open protocol for agent-to-agent communication. This package ships a
stdlib-only client (JSON-RPC over HTTP) plus :class:`DelegateNode`, a graph
node that hands a task to a remote agent and merges its outcome back into
state — including input-required negotiation via pyantra's interrupt
machinery.

Everything is dependency-free; there is no optional extra to install.
"""

from pyantra.a2a.client import A2aClient, A2aClientProtocol
from pyantra.a2a.errors import A2aError
from pyantra.a2a.node import DelegateNode
from pyantra.a2a.types import (
    AgentCard,
    DataPart,
    FilePart,
    Message,
    Part,
    Task,
    TaskStatus,
    TextPart,
    part_from_dict,
)

__all__ = [
    "A2aClient",
    "A2aClientProtocol",
    "A2aError",
    "AgentCard",
    "DataPart",
    "DelegateNode",
    "FilePart",
    "Message",
    "Part",
    "Task",
    "TaskStatus",
    "TextPart",
    "part_from_dict",
]
