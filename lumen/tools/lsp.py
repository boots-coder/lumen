"""
LSP tool — code intelligence via Language Server Protocol.

Provides go-to-definition, find-references, hover info, and symbol search
by talking to an appropriate language server. Gracefully degrades when no
server is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult
from ..services.lsp import (
    LSPClient,
    Location,
    Symbol,
    get_client,
    language_from_path,
)

logger = logging.getLogger(__name__)


# ── Input schema ─────────────────────────────────────────────────────────────

class LSPInput(BaseModel):
    action: str = Field(
        description=(
            "One of: definition, references, hover, symbols, workspace_symbols"
        ),
    )
    file_path: str = Field(
        description="Absolute file path for the operation",
    )
    line: int | None = Field(
        default=None,
        description="Line number (1-based) — required for definition/references/hover",
    )
    column: int | None = Field(
        default=None,
        description="Column number (1-based) — required for definition/references/hover",
    )
    query: str | None = Field(
        default=None,
        description="Query string for workspace_symbols action",
    )


# ── Formatting helpers ───────────────────────────────────────────────────────

def _format_locations(locations: list[Location], label: str) -> str:
    if not locations:
        return f"No {label} found."
    if len(locations) == 1:
        return f"{label.capitalize()} found at: {locations[0]}"
    lines = [f"{len(locations)} {label} found:"]
    for i, loc in enumerate(locations, 1):
        lines.append(f"  {i}. {loc}")
    return "\n".join(lines)


def _format_symbols(symbols: list[Symbol], context: str) -> str:
    if not symbols:
        return f"No symbols found in {context}."
    lines = [f"{len(symbols)} symbol(s) in {context}:"]
    for sym in symbols:
        container = f"{sym.container_name}." if sym.container_name else ""
        lines.append(f"  {sym.kind}: {container}{sym.name} (line {sym.location.line})")
    return "\n".join(lines)


# ── Tool ─────────────────────────────────────────────────────────────────────

class LSPTool(Tool[LSPInput]):
    """
    Code intelligence: go-to-definition, find references, hover info, symbol search.

    Uses Language Server Protocol to provide accurate, semantic code navigation.
    Requires an LSP server for the target language to be installed on the system.

    Supported languages: Python (pylsp/pyright), TypeScript/JS, Go (gopls), Rust (rust-analyzer).
    """

    @property
    def name(self) -> str:
        return "lsp"

    @property
    def description(self) -> str:
        return (
            "Code intelligence: go-to-definition, find references, hover info, "
            "symbol search. Uses Language Server Protocol.\n\n"
            "Actions:\n"
            "  definition        — jump to the definition of a symbol\n"
            "  references        — find all references to a symbol\n"
            "  hover             — get type/doc info for a symbol\n"
            "  symbols           — list all symbols in a file\n"
            "  workspace_symbols — search symbols across the project\n\n"
            "For definition/references/hover, provide file_path + line + column (1-based).\n"
            "For symbols, provide file_path.\n"
            "For workspace_symbols, provide file_path (to detect language) + query.\n\n"
            'Example: {"action": "definition", "file_path": "/path/to/file.py", "line": 42, "column": 10}\n'
            'Example: {"action": "symbols", "file_path": "/path/to/file.py"}\n'
            'Example: {"action": "workspace_symbols", "file_path": "/path/to/file.py", "query": "MyClass"}'
        )

    @property
    def input_schema(self) -> type[LSPInput]:
        return LSPInput

    async def execute(self, input_data: LSPInput) -> ToolResult:
        action = input_data.action.lower().strip()
        file_path = input_data.file_path

        # Validate action
        valid_actions = {"definition", "references", "hover", "symbols", "workspace_symbols"}
        if action not in valid_actions:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown action '{action}'. Must be one of: {', '.join(sorted(valid_actions))}",
            )

        # Validate file path
        path = Path(file_path).expanduser().resolve()
        if action != "workspace_symbols" and not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path}",
            )

        # Detect language
        language = language_from_path(str(path))
        if language is None:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Unsupported file type: {path.suffix}. "
                    "LSP supports: .py, .ts/.tsx, .js/.jsx, .go, .rs"
                ),
            )

        # Validate line/column for positional actions
        if action in ("definition", "references", "hover"):
            if input_data.line is None or input_data.column is None:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Action '{action}' requires both line and column (1-based).",
                )

        # Validate query for workspace_symbols
        if action == "workspace_symbols" and not input_data.query:
            return ToolResult(
                success=False,
                output="",
                error="Action 'workspace_symbols' requires a query string.",
            )

        # Get or create LSP client
        try:
            root_path = self._find_project_root(str(path))
            client = await get_client(language, root_path)
        except RuntimeError as exc:
            return ToolResult(
                success=False,
                output="",
                error=str(exc),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to start LSP server for {language}: {exc}",
            )

        # Dispatch to the appropriate action
        try:
            if action == "definition":
                assert input_data.line is not None and input_data.column is not None
                locations = await client.goto_definition(
                    str(path), input_data.line, input_data.column,
                )
                output = _format_locations(locations, "definition")

            elif action == "references":
                assert input_data.line is not None and input_data.column is not None
                locations = await client.find_references(
                    str(path), input_data.line, input_data.column,
                )
                output = _format_locations(locations, "references")

            elif action == "hover":
                assert input_data.line is not None and input_data.column is not None
                hover_text = await client.hover(
                    str(path), input_data.line, input_data.column,
                )
                if hover_text:
                    output = f"Hover info:\n{hover_text}"
                else:
                    output = "No hover information available at this position."

            elif action == "symbols":
                symbols = await client.document_symbols(str(path))
                output = _format_symbols(symbols, path.name)

            elif action == "workspace_symbols":
                assert input_data.query is not None
                symbols = await client.workspace_symbols(input_data.query)
                output = _format_symbols(symbols, f"workspace (query: '{input_data.query}')")

            else:
                output = ""  # unreachable

            return ToolResult(success=True, output=output)

        except Exception as exc:
            logger.debug("LSP action %s failed: %s", action, exc)
            return ToolResult(
                success=False,
                output="",
                error=f"LSP {action} failed: {exc}",
            )

    @staticmethod
    def _find_project_root(file_path: str) -> str:
        """Walk up from file to find project root (directory with .git, pyproject.toml, etc.)."""
        markers = {
            ".git", "pyproject.toml", "setup.py", "setup.cfg",
            "package.json", "go.mod", "Cargo.toml",
            ".project", "pom.xml", "build.gradle",
        }
        current = Path(file_path).resolve()
        if current.is_file():
            current = current.parent

        for directory in [current, *current.parents]:
            if any((directory / marker).exists() for marker in markers):
                return str(directory)

        # Fallback to file's parent directory
        return str(Path(file_path).resolve().parent)
