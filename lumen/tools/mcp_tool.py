"""
MCPToolBridge — expose a tool advertised by an MCP server as a lumen Tool.

Key trick: MCP tools announce their inputs via raw JSON Schema. Instead of
round-tripping that through a synthesized Pydantic model, we:

  · use a permissive Pydantic `ConfigDict(extra='allow')` input model so the
    dispatcher's `tool.input_schema(**arguments)` call accepts any kwargs
  · override `to_openai_schema()` / `to_anthropic_schema()` to emit the MCP
    server's own JSON Schema directly, so the LLM sees the real tool shape

On invocation we forward the raw kwargs to `client.call_tool(...)` and flatten
the MCP `content[]` list back into plain text.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from ..services.mcp.client import MCPClient, MCPTool


class _MCPPassthrough(BaseModel):
    """Accepts any kwargs; we read them back via model_dump()."""
    model_config = ConfigDict(extra="allow")


class MCPToolBridge(Tool[_MCPPassthrough]):
    """Wrap a single tool from an MCP server."""

    def __init__(
        self,
        server_name: str,
        client: MCPClient,
        mcp_tool: MCPTool,
    ) -> None:
        self._server_name = server_name
        self._client = client
        self._mcp_tool = mcp_tool
        # Scoped name so two servers announcing the same tool don't collide
        self._local_name = f"mcp__{server_name}__{mcp_tool.name}"

    @property
    def name(self) -> str:
        return self._local_name

    @property
    def description(self) -> str:
        return (
            self._mcp_tool.description
            or f"{self._server_name}: {self._mcp_tool.name} (MCP tool)"
        )

    @property
    def input_schema(self) -> type[_MCPPassthrough]:
        return _MCPPassthrough

    @property
    def mcp_tool_name(self) -> str:
        """Original name as advertised by the server (without our prefix)."""
        return self._mcp_tool.name

    @property
    def server_name(self) -> str:
        return self._server_name

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._local_name,
                "description": self.description,
                "parameters": self._mcp_tool.input_schema,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self._local_name,
            "description": self.description,
            "input_schema": self._mcp_tool.input_schema,
        }

    async def execute(self, input_data: _MCPPassthrough) -> ToolResult:
        arguments = input_data.model_dump()
        try:
            result = await self._client.call_tool(
                self._mcp_tool.name, arguments,
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

        # MCP returns {"content": [{"type": "text", "text": "..."}, ...],
        #              "isError": bool}
        is_error = bool(result.get("isError", False))
        text = _flatten_content(result.get("content", []))
        return ToolResult(
            success=not is_error,
            output=text,
            error=text if is_error else None,
        )


def _flatten_content(items: list[dict[str, Any]]) -> str:
    """Turn MCP tool result content[] into a single string."""
    chunks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            chunks.append(item.get("text", ""))
        elif t == "image":
            chunks.append(f"[image: {item.get('mimeType', 'unknown')}]")
        elif t == "resource":
            res = item.get("resource", {})
            chunks.append(f"[resource: {res.get('uri', 'unknown')}]")
        else:
            chunks.append(json.dumps(item))
    return "\n".join(chunks)
