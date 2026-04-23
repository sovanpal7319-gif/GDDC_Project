"""
core/mcp.py
Model Context Protocol (MCP) — lightweight message-passing layer between agents.
Each agent wraps its output in an MCPMessage and broadcasts via the MCPBus.
Agents can subscribe to message types and receive them asynchronously.
"""
from __future__ import annotations
import asyncio
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class MCPMessage:
    """A message passed between agents via the MCP bus."""
    sender: str                    # Agent name/ID
    message_type: str              # e.g. "sentiment_result", "timeseries_result"
    payload: Any                   # The actual data (Pydantic model or dict)
    query_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# Type alias for async message handlers
Handler = Callable[[MCPMessage], Awaitable[None]]


class MCPBus:
    """
    In-process async pub/sub message bus.
    In production this could be replaced with Redis Streams, NATS, or Kafka.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Handler]] = {}
        self._message_log: list[MCPMessage] = []

    def subscribe(self, message_type: str, handler: Handler) -> None:
        self._subscribers.setdefault(message_type, []).append(handler)
        logger.debug(f"MCP: subscribed to '{message_type}'")

    async def publish(self, message: MCPMessage) -> None:
        self._message_log.append(message)
        logger.info(
            f"MCP: {message.sender} → '{message.message_type}' "
            f"(query={message.query_id})"
        )
        handlers = self._subscribers.get(message.message_type, [])
        if handlers:
            await asyncio.gather(*[h(message) for h in handlers])

    def get_log(self) -> list[MCPMessage]:
        return list(self._message_log)


# Singleton bus instance shared across agents
mcp_bus = MCPBus()
