"""Per-node execution timeouts."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


async def with_timeout(coro: Coroutine[Any, Any, T], timeout: float) -> T:
    """Run ``coro`` with a timeout in seconds.

    On timeout the underlying task is cancelled. Cancellation only interrupts
    cooperative (await-based) code, so a node performing synchronous blocking
    work is not interrupted until it returns control to the event loop.
    """
    task = asyncio.create_task(coro)
    return await asyncio.wait_for(task, timeout)
