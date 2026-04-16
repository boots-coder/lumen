"""
FileWrite tool — create new files or completely overwrite existing ones.

Design decisions:
- Creates parent directories automatically
- Refuses to overwrite without explicit flag (safety)
- Size guard: max 500KB per write
- UTF-8 encoding only
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult


class FileWriteInput(BaseModel):
    """Input schema for FileWrite tool."""

    file_path: str = Field(
        description="Absolute path to the file to write."
    )
    content: str = Field(
        description="The full content to write to the file."
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "If True, overwrite existing file. "
            "If False (default), refuse to overwrite an existing file. "
            "Set to True only when you intend to completely replace a file."
        ),
    )


class FileWriteTool(Tool[FileWriteInput]):
    """
    Write content to a file. Creates parent directories if needed.

    Use this to create new files. For modifying existing files,
    prefer the edit_file tool (surgical find-and-replace) over
    overwriting the entire file.
    """

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file (create new or overwrite existing).\n\n"
            "Creates parent directories automatically if they don't exist.\n\n"
            "IMPORTANT:\n"
            "  - For NEW files: just provide file_path and content.\n"
            "  - For EXISTING files: you MUST set overwrite=true.\n"
            "  - Prefer edit_file for surgical modifications to existing files.\n"
            "  - Only use write_file to overwrite when the entire file needs replacing.\n\n"
            "Parameters:\n"
            "  file_path  — absolute path to write to (required)\n"
            "  content    — full file content (required)\n"
            "  overwrite  — set true to overwrite existing file (default false)\n\n"
            "Examples:\n"
            '  New file:       {"file_path": "/abs/path/new.py", "content": "print(\'hello\')"}\n'
            '  Overwrite:      {"file_path": "/abs/path/old.py", "content": "...", "overwrite": true}'
        )

    @property
    def input_schema(self) -> type[FileWriteInput]:
        return FileWriteInput

    async def execute(self, input_data: FileWriteInput) -> ToolResult:
        try:
            path = Path(input_data.file_path).expanduser().resolve()

            # Safety: refuse to write to certain paths
            dangerous = {"/etc", "/usr", "/bin", "/sbin", "/var", "/System", "/Library"}
            for d in dangerous:
                if str(path).startswith(d):
                    return ToolResult(
                        success=False, output="",
                        error=f"Refusing to write to system directory: {d}",
                    )

            # Size guard (500 KB)
            content_bytes = input_data.content.encode("utf-8")
            if len(content_bytes) > 500 * 1024:
                return ToolResult(
                    success=False, output="",
                    error="Content exceeds 500 KB limit. Break into smaller files.",
                )

            # Check overwrite safety
            if path.exists() and not input_data.overwrite:
                return ToolResult(
                    success=False, output="",
                    error=(
                        f"File already exists: {input_data.file_path}\n"
                        "Set overwrite=true to replace it, or use edit_file for surgical edits."
                    ),
                )

            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            path.write_text(input_data.content, encoding="utf-8")

            line_count = input_data.content.count("\n") + (1 if input_data.content else 0)
            action = "overwritten" if path.exists() and input_data.overwrite else "created"

            return ToolResult(
                success=True,
                output=(
                    f"[write_file] Successfully {action}: {path}\n"
                    f"  {line_count} lines, {len(content_bytes)} bytes written."
                ),
            )

        except PermissionError:
            return ToolResult(
                success=False, output="",
                error=f"Permission denied: {input_data.file_path}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"write_file error: {e}")
