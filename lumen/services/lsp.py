"""
LSP Integration — code intelligence via Language Server Protocol.

Provides:
  - Go to definition
  - Find references
  - Hover information (type info, docs)
  - Document symbols
  - Workspace symbol search

Design: manages LSP server lifecycle, communicates via JSON-RPC over stdio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class Location:
    """A source-code location returned by LSP."""

    file_path: str
    line: int          # 1-based
    column: int        # 1-based
    end_line: int | None = None
    end_column: int | None = None

    def __str__(self) -> str:
        s = f"{self.file_path}:{self.line}:{self.column}"
        if self.end_line is not None and self.end_line != self.line:
            s += f"-{self.end_line}"
        return s


@dataclass
class Symbol:
    """A symbol (class, function, variable, etc.) found via LSP."""

    name: str
    kind: str           # function, class, variable, method, …
    location: Location
    container_name: str | None = None

    def __str__(self) -> str:
        prefix = f"{self.container_name}." if self.container_name else ""
        return f"{self.kind}: {prefix}{self.name} (line {self.location.line})"


# ── LSP symbol-kind mapping ─────────────────────────────────────────────────

_SYMBOL_KIND_MAP: dict[int, str] = {
    1: "file", 2: "module", 3: "namespace", 4: "package",
    5: "class", 6: "method", 7: "property", 8: "field",
    9: "constructor", 10: "enum", 11: "interface", 12: "function",
    13: "variable", 14: "constant", 15: "string", 16: "number",
    17: "boolean", 18: "array", 19: "object", 20: "key",
    21: "null", 22: "enum_member", 23: "struct", 24: "event",
    25: "operator", 26: "type_parameter",
}


# ── Known language-server commands ───────────────────────────────────────────

_SERVER_COMMANDS: dict[str, list[list[str]]] = {
    "python": [
        ["pylsp"],
        ["pyright-langserver", "--stdio"],
    ],
    "typescript": [
        ["typescript-language-server", "--stdio"],
    ],
    "javascript": [
        ["typescript-language-server", "--stdio"],
    ],
    "go": [
        ["gopls", "serve"],
    ],
    "rust": [
        ["rust-analyzer"],
    ],
}

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
}

_LANGUAGE_IDS: dict[str, str] = {
    "python": "python",
    "typescript": "typescriptreact",
    "javascript": "javascriptreact",
    "go": "go",
    "rust": "rust",
}


def language_from_path(path: str) -> str | None:
    """Detect language from file extension."""
    ext = Path(path).suffix.lower()
    return _EXT_TO_LANGUAGE.get(ext)


def _find_server_command(language: str) -> list[str] | None:
    """Find the first available LSP server for the given language."""
    candidates = _SERVER_COMMANDS.get(language, [])
    for cmd in candidates:
        if shutil.which(cmd[0]) is not None:
            return cmd
    return None


# ── JSON-RPC helpers ─────────────────────────────────────────────────────────

def _encode_message(obj: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message with Content-Length header."""
    body = json.dumps(obj).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


async def _read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one JSON-RPC message from the stream. Returns None on EOF."""
    # Read headers
    content_length: int | None = None
    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            return None  # EOF
        line = line_bytes.decode("ascii", errors="replace").strip()
        if not line:
            break  # end of headers
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    if content_length is None:
        return None

    body_bytes = await reader.readexactly(content_length)
    return json.loads(body_bytes)


# ── LSP Client ───────────────────────────────────────────────────────────────

class LSPClient:
    """
    Lightweight LSP client that talks to a language server over stdio.

    Usage::

        client = LSPClient("python")
        await client.start("/path/to/project")
        locs = await client.goto_definition("file.py", 10, 5)
        await client.shutdown()
    """

    def __init__(
        self,
        language: str,
        server_command: list[str] | None = None,
    ) -> None:
        self.language = language
        self._server_command = server_command
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None  # not used; we write to stdin
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._initialized = False
        self._root_uri = ""
        self._open_files: set[str] = set()
        self._timeout = 15.0  # seconds per request

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self, root_path: str) -> None:
        """Spawn the language server and send initialize + initialized."""
        if self._initialized:
            return

        cmd = self._server_command or _find_server_command(self.language)
        if cmd is None:
            raise RuntimeError(
                f"No LSP server found for {self.language}. "
                f"Install one of: {', '.join(c[0] for c in _SERVER_COMMANDS.get(self.language, []))}"
            )

        self._root_uri = Path(root_path).resolve().as_uri()

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=root_path,
        )

        self._reader = self._process.stdout
        self._reader_task = asyncio.create_task(self._read_loop())

        # Send initialize
        init_result = await self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": self._root_uri,
            "rootPath": str(Path(root_path).resolve()),
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                },
                "workspace": {
                    "symbol": {"dynamicRegistration": False},
                },
            },
            "workspaceFolders": [
                {"uri": self._root_uri, "name": Path(root_path).name},
            ],
        })

        # Send initialized notification
        self._notify("initialized", {})
        self._initialized = True

    async def shutdown(self) -> None:
        """Cleanly shut down the language server."""
        if not self._initialized:
            return
        try:
            await self._request("shutdown", None, timeout=5.0)
            self._notify("exit", None)
        except Exception:
            pass
        finally:
            self._initialized = False
            if self._reader_task and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self._process:
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=3.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
            # Cancel any remaining pending futures
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()
            self._open_files.clear()

    # ── public API ───────────────────────────────────────────────────────

    async def goto_definition(self, file: str, line: int, col: int) -> list[Location]:
        """Go to definition. line/col are 1-based."""
        await self._ensure_open(file)
        result = await self._request("textDocument/definition", {
            "textDocument": {"uri": Path(file).resolve().as_uri()},
            "position": {"line": line - 1, "character": col - 1},
        })
        return self._parse_locations(result)

    async def find_references(self, file: str, line: int, col: int) -> list[Location]:
        """Find all references. line/col are 1-based."""
        await self._ensure_open(file)
        result = await self._request("textDocument/references", {
            "textDocument": {"uri": Path(file).resolve().as_uri()},
            "position": {"line": line - 1, "character": col - 1},
            "context": {"includeDeclaration": True},
        })
        return self._parse_locations(result)

    async def hover(self, file: str, line: int, col: int) -> str | None:
        """Get hover information. line/col are 1-based."""
        await self._ensure_open(file)
        result = await self._request("textDocument/hover", {
            "textDocument": {"uri": Path(file).resolve().as_uri()},
            "position": {"line": line - 1, "character": col - 1},
        })
        if not result:
            return None
        contents = result.get("contents")
        return self._extract_hover_text(contents)

    async def document_symbols(self, file: str) -> list[Symbol]:
        """Get all symbols in a document."""
        await self._ensure_open(file)
        result = await self._request("textDocument/documentSymbol", {
            "textDocument": {"uri": Path(file).resolve().as_uri()},
        })
        if not result:
            return []
        return self._parse_symbols(result, file)

    async def workspace_symbols(self, query: str) -> list[Symbol]:
        """Search symbols across the workspace."""
        result = await self._request("workspace/symbol", {"query": query})
        if not result:
            return []
        return self._parse_workspace_symbols(result)

    # ── file management ──────────────────────────────────────────────────

    async def _ensure_open(self, file: str) -> None:
        """Send textDocument/didOpen if the file hasn't been opened yet."""
        resolved = str(Path(file).resolve())
        if resolved in self._open_files:
            return
        try:
            text = Path(resolved).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        lang_id = _LANGUAGE_IDS.get(self.language, self.language)
        self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": Path(resolved).as_uri(),
                "languageId": lang_id,
                "version": 1,
                "text": text,
            },
        })
        self._open_files.add(resolved)
        # Give the server a moment to process the file
        await asyncio.sleep(0.1)

    # ── JSON-RPC transport ───────────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _request(
        self,
        method: str,
        params: Any,
        timeout: float | None = None,
    ) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("LSP server not started")

        req_id = self._next_id()
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        data = _encode_message(msg)
        self._process.stdin.write(data)
        await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(fut, timeout=timeout or self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            logger.warning("LSP request timed out: %s", method)
            return None

    def _notify(self, method: str, params: Any) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self._process is None or self._process.stdin is None:
            return
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        data = _encode_message(msg)
        self._process.stdin.write(data)
        # Don't await drain for notifications — fire and forget

    async def _read_loop(self) -> None:
        """Background task that reads JSON-RPC messages from the server."""
        assert self._reader is not None
        try:
            while True:
                msg = await _read_message(self._reader)
                if msg is None:
                    break  # EOF

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        if "error" in msg:
                            err = msg["error"]
                            logger.debug(
                                "LSP error (id=%s): [%s] %s",
                                msg_id, err.get("code"), err.get("message"),
                            )
                            fut.set_result(None)
                        else:
                            fut.set_result(msg.get("result"))
                # else: notification or server request — ignore
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("LSP read loop error: %s", exc)

    # ── response parsing ─────────────────────────────────────────────────

    @staticmethod
    def _uri_to_path(uri: str) -> str:
        """Convert file:// URI to local path."""
        if uri.startswith("file://"):
            from urllib.parse import unquote, urlparse
            parsed = urlparse(uri)
            return unquote(parsed.path)
        return uri

    def _parse_location(self, loc: dict[str, Any]) -> Location:
        """Parse a single LSP Location or LocationLink."""
        # LocationLink has targetUri / targetRange
        if "targetUri" in loc:
            uri = loc["targetUri"]
            rng = loc.get("targetSelectionRange") or loc.get("targetRange", {})
        else:
            uri = loc.get("uri", "")
            rng = loc.get("range", {})

        start = rng.get("start", {})
        end = rng.get("end", {})

        return Location(
            file_path=self._uri_to_path(uri),
            line=start.get("line", 0) + 1,
            column=start.get("character", 0) + 1,
            end_line=end.get("line", 0) + 1 if end else None,
            end_column=end.get("character", 0) + 1 if end else None,
        )

    def _parse_locations(self, result: Any) -> list[Location]:
        """Parse goto-definition / find-references results."""
        if result is None:
            return []
        if isinstance(result, dict):
            return [self._parse_location(result)]
        if isinstance(result, list):
            return [self._parse_location(loc) for loc in result if isinstance(loc, dict)]
        return []

    @staticmethod
    def _extract_hover_text(contents: Any) -> str | None:
        """Extract readable text from hover contents."""
        if contents is None:
            return None
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            value = contents.get("value", "")
            return value if value else None
        if isinstance(contents, list):
            parts: list[str] = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", ""))
            text = "\n".join(p for p in parts if p)
            return text if text else None
        return None

    def _parse_symbols(
        self,
        result: list[Any],
        file: str,
        container: str | None = None,
    ) -> list[Symbol]:
        """Parse DocumentSymbol[] (hierarchical) or SymbolInformation[]."""
        symbols: list[Symbol] = []
        for item in result:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "")
            kind_num = item.get("kind", 0)
            kind_str = _SYMBOL_KIND_MAP.get(kind_num, f"kind({kind_num})")

            # DocumentSymbol has selectionRange; SymbolInformation has location
            if "selectionRange" in item:
                rng = item["selectionRange"]
                start = rng.get("start", {})
                loc = Location(
                    file_path=file,
                    line=start.get("line", 0) + 1,
                    column=start.get("character", 0) + 1,
                )
            elif "location" in item:
                loc_data = item["location"]
                start = loc_data.get("range", {}).get("start", {})
                loc = Location(
                    file_path=self._uri_to_path(loc_data.get("uri", file)),
                    line=start.get("line", 0) + 1,
                    column=start.get("character", 0) + 1,
                )
            else:
                loc = Location(file_path=file, line=1, column=1)

            container_name = container or item.get("containerName")
            symbols.append(Symbol(name, kind_str, loc, container_name))

            # Recurse into children (DocumentSymbol)
            children = item.get("children")
            if children:
                symbols.extend(self._parse_symbols(children, file, container=name))

        return symbols

    def _parse_workspace_symbols(self, result: list[Any]) -> list[Symbol]:
        """Parse SymbolInformation[] from workspace/symbol."""
        symbols: list[Symbol] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            kind_num = item.get("kind", 0)
            kind_str = _SYMBOL_KIND_MAP.get(kind_num, f"kind({kind_num})")

            loc_data = item.get("location", {})
            uri = loc_data.get("uri", "")
            start = loc_data.get("range", {}).get("start", {})

            loc = Location(
                file_path=self._uri_to_path(uri),
                line=start.get("line", 0) + 1,
                column=start.get("character", 0) + 1,
            )
            symbols.append(Symbol(name, kind_str, loc, item.get("containerName")))
        return symbols


# ── Client cache ─────────────────────────────────────────────────────────────

_clients: dict[str, LSPClient] = {}


async def get_client(
    language: str,
    root_path: str,
    server_command: list[str] | None = None,
) -> LSPClient:
    """Get or create a cached LSPClient for the given language."""
    if language in _clients and _clients[language]._initialized:
        return _clients[language]
    client = LSPClient(language, server_command)
    await client.start(root_path)
    _clients[language] = client
    return client


async def shutdown_all() -> None:
    """Shut down all cached LSP clients."""
    for client in _clients.values():
        await client.shutdown()
    _clients.clear()
