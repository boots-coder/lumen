"""
Minimal MCP client (stdio transport).

Speaks JSON-RPC 2.0 over the subprocess's stdin/stdout, using the
line-delimited framing that every MCP server implements.

  · `initialize`           — handshake, get server capabilities
  · `tools/list`           — discover tools
  · `tools/call`           — invoke a tool

Not yet implemented (future waves): resources, prompts, sampling,
notifications pushed by the server (subscriptions), websocket transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "lumen"
CLIENT_VERSION = "0.5.0"


class MCPError(RuntimeError):
    """Raised on protocol errors, timeouts, or server-returned errors."""


@dataclass
class MCPTool:
    """A tool advertised by an MCP server."""

    name: str                    # tool name (as announced by server)
    description: str
    input_schema: dict[str, Any]  # raw JSON Schema object


# ─────────────────────────────────────────────────────────────────────────────

class MCPClient:
    """
    stdio JSON-RPC client for a single MCP server.

    Lifecycle: `start()` → `initialize()` → repeated `list_tools()` /
    `call_tool()` → `close()`.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._cwd = cwd
        self._timeout = request_timeout

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._initialized = False
        self._server_info: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}
        self._send_lock = asyncio.Lock()
        self._stderr_buf: list[str] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._proc is not None:
            raise MCPError("Already started")
        merged_env = {**os.environ, **self._env}
        logger.debug("Spawning MCP server: %s %s", self._command, " ".join(self._args))
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._command, *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
                cwd=self._cwd,
            )
        except FileNotFoundError as e:
            raise MCPError(f"MCP command not found: {self._command}") from e

        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"mcp-reader-{self._command}",
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_loop(), name=f"mcp-stderr-{self._command}",
        )

    async def initialize(self) -> dict[str, Any]:
        """Perform the MCP handshake. Returns server info."""
        if self._initialized:
            return self._server_info
        result = await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
        })
        self._server_info = result.get("serverInfo", {})
        self._capabilities = result.get("capabilities", {})
        # The spec requires a `notifications/initialized` after the handshake.
        await self._notify("notifications/initialized", {})
        self._initialized = True
        return self._server_info

    async def close(self) -> None:
        if self._proc is None:
            return
        # Fail all pending requests so callers unblock
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError("Connection closed"))
        self._pending.clear()

        try:
            if self._proc.returncode is None:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
        except ProcessLookupError:
            pass

        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        self._proc = None
        self._reader_task = None
        self._stderr_task = None

    # ── high-level API ───────────────────────────────────────────────────────

    async def list_tools(self) -> list[MCPTool]:
        result = await self._request("tools/list", {})
        tools_raw = result.get("tools", [])
        return [
            MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
            )
            for t in tools_raw
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def recent_stderr(self) -> str:
        """Last 30 stderr lines — useful to surface when the server crashes."""
        return "\n".join(self._stderr_buf[-30:])

    # ── low-level JSON-RPC ───────────────────────────────────────────────────

    def _allocate_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def _send_raw(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MCPError("Not started")
        data = (json.dumps(message) + "\n").encode("utf-8")
        async with self._send_lock:
            try:
                self._proc.stdin.write(data)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                raise MCPError(f"Connection broken: {e}") from e

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        req_id = self._allocate_id()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self._send_raw({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })
        try:
            return await asyncio.wait_for(fut, timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPError(
                f"Timeout waiting for {method!r} response "
                f"(stderr tail: {self.recent_stderr[-200:]})"
            )

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send_raw({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })

    async def _read_loop(self) -> None:
        """Dispatch incoming JSON-RPC messages to waiting futures."""
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while True:
            try:
                line = await stdout.readline()
            except Exception as e:
                logger.warning("MCP read error: %s", e)
                break
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from MCP: %r", line[:200])
                continue
            self._handle_message(msg)

        # Reader exited — fail pending requests so callers unblock
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError(
                    "Server closed stdout (process likely exited)"
                ))
        self._pending.clear()

    def _handle_message(self, msg: dict[str, Any]) -> None:
        # Response to one of our requests
        if "id" in msg and ("result" in msg or "error" in msg):
            fut = self._pending.pop(msg["id"], None)
            if fut is None:
                return
            if "error" in msg:
                err = msg["error"]
                fut.set_exception(MCPError(
                    f"{err.get('code', '?')} {err.get('message', 'unknown')}"
                ))
            else:
                fut.set_result(msg.get("result"))
            return
        # Server-initiated request or notification — we don't handle these yet
        logger.debug("Unhandled server message: %s", msg)

    async def _stderr_loop(self) -> None:
        """Buffer the last N stderr lines for error context."""
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            try:
                line = await self._proc.stderr.readline()
            except Exception:
                break
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            self._stderr_buf.append(decoded)
            if len(self._stderr_buf) > 100:
                self._stderr_buf = self._stderr_buf[-50:]
