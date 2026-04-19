"""
MCPManager — connect to one or many MCP servers, discover their tools,
expose them as lumen Tool instances.

Config format (`~/.lumen/mcp.json`), compatible with Claude Desktop's
`claude_desktop_config.json`:

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        },
        "github": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "env": {"GITHUB_TOKEN": "..."},
          "disabled": true
        }
      }
    }

Entries with `"disabled": true` are skipped. Missing commands are reported
rather than raised, so one bad entry doesn't block startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .client import MCPClient, MCPError, MCPTool

if TYPE_CHECKING:
    from ...tools.base import Tool

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    disabled: bool = False


@dataclass
class MCPServerState:
    config: MCPServerConfig
    client: MCPClient | None = None
    tools: list[MCPTool] = field(default_factory=list)
    status: str = "idle"   # idle | connecting | connected | failed | closed
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────

class MCPManager:
    """
    Orchestrates MCP servers. Connect once, manager holds clients + caches
    discovered tools. Call `tools()` to get bridge wrappers ready to register
    on an Agent.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerState] = {}

    # ── config ───────────────────────────────────────────────────────────────

    @staticmethod
    def parse_config(data: dict[str, Any]) -> list[MCPServerConfig]:
        servers_raw = data.get("mcpServers") or data.get("servers") or {}
        out: list[MCPServerConfig] = []
        for name, spec in servers_raw.items():
            if not isinstance(spec, dict) or "command" not in spec:
                logger.warning("Skipping malformed MCP server %r", name)
                continue
            out.append(MCPServerConfig(
                name=name,
                command=spec["command"],
                args=list(spec.get("args", [])),
                env=dict(spec.get("env", {})),
                cwd=spec.get("cwd"),
                disabled=bool(spec.get("disabled", False)),
            ))
        return out

    def load_config(self, path: Path) -> list[MCPServerConfig]:
        path = Path(path).expanduser()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("Invalid MCP config at %s: %s", path, e)
            return []
        return self.parse_config(data)

    # ── connect / disconnect ─────────────────────────────────────────────────

    async def connect(
        self,
        config: MCPServerConfig,
        timeout: float = 15.0,
    ) -> MCPServerState:
        if config.disabled:
            state = MCPServerState(config=config, status="disabled")
            self._servers[config.name] = state
            return state

        state = MCPServerState(config=config, status="connecting")
        self._servers[config.name] = state

        client = MCPClient(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
            request_timeout=timeout,
        )
        try:
            await client.start()
            await client.initialize()
            state.tools = await client.list_tools()
            state.client = client
            state.status = "connected"
            logger.info(
                "MCP %s connected — %d tool(s): %s",
                config.name, len(state.tools),
                ", ".join(t.name for t in state.tools[:8]),
            )
        except (MCPError, asyncio.TimeoutError, OSError) as e:
            await client.close()
            state.status = "failed"
            state.error = str(e) or client.recent_stderr or type(e).__name__
            logger.warning("MCP %s failed: %s", config.name, state.error)
        return state

    async def connect_all(
        self,
        configs: list[MCPServerConfig],
        timeout: float = 15.0,
    ) -> list[MCPServerState]:
        return await asyncio.gather(*(
            self.connect(c, timeout=timeout) for c in configs
        ))

    async def disconnect(self, name: str) -> bool:
        state = self._servers.get(name)
        if state is None or state.client is None:
            return False
        await state.client.close()
        state.client = None
        state.status = "closed"
        return True

    async def close(self) -> None:
        await asyncio.gather(*(
            self.disconnect(name) for name in list(self._servers)
        ), return_exceptions=True)

    # ── queries ──────────────────────────────────────────────────────────────

    def servers(self) -> list[MCPServerState]:
        return list(self._servers.values())

    def tools(self) -> list[Tool]:
        """Return lumen Tool bridges for every connected server's tools."""
        from ...tools.mcp_tool import MCPToolBridge
        bridges: list[Tool] = []
        for state in self._servers.values():
            if state.client is None or state.status != "connected":
                continue
            for t in state.tools:
                bridges.append(MCPToolBridge(
                    server_name=state.config.name,
                    client=state.client,
                    mcp_tool=t,
                ))
        return bridges

    def get(self, name: str) -> MCPServerState | None:
        return self._servers.get(name)

    def __len__(self) -> int:
        return len(self._servers)
