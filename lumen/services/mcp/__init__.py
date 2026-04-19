"""
MCP (Model Context Protocol) client — connect lumen to any MCP-compatible
server and surface its tools to the model.

Usage:

    from lumen import Agent
    from lumen.services.mcp import MCPManager, MCPServerConfig

    mgr = MCPManager()
    await mgr.connect(MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    ))
    # All discovered tools are exposed as `mcp_{server}__{tool}` and can
    # be registered onto an Agent:
    for bridge in mgr.tools():
        agent._tool_registry.register(bridge)

Config file support: the manager can load `~/.lumen/mcp.json` via
`MCPManager.load_config(path)`.
"""

from .client import MCPClient, MCPError, MCPTool
from .manager import MCPManager, MCPServerConfig, MCPServerState

__all__ = [
    "MCPClient",
    "MCPError",
    "MCPTool",
    "MCPManager",
    "MCPServerConfig",
    "MCPServerState",
]
