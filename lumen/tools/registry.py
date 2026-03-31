"""Tool registry for managing available tools."""

from __future__ import annotations

import logging
from typing import Any

from .base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing tools available to the agent.

    Handles tool registration, lookup, and schema conversion.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI function calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to Anthropic tool use format."""
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools: {', '.join(self.list_tools())})"
