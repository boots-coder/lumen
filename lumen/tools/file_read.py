"""
FileRead tool — read file contents with line-range support.

Key design decisions:
- Default limit: 2000 lines  (prevents accidental huge reads)
- offset is 1-indexed        (matches line numbers shown in output)
- cat -n style output        (spaces + line_number + tab + content)
- Truncation notice          (tells model more lines exist and how to page)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

_DEFAULT_LIMIT = 2000

# Binary-ish extensions we refuse to read as text
_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv",
    ".db", ".sqlite", ".sqlite3",
}


class FileReadInput(BaseModel):
    """Input schema for FileRead tool."""

    file_path: str = Field(
        description="Absolute path to the file to read."
    )
    offset: int | None = Field(
        None,
        description=(
            "Line number to start reading from (1-indexed, inclusive). "
            "Use together with limit to page through large files."
        ),
    )
    limit: int | None = Field(
        None,
        description=(
            f"Maximum number of lines to read. "
            f"Defaults to {_DEFAULT_LIMIT}. "
            "Increase only when you need more context."
        ),
    )


class FileReadTool(Tool[FileReadInput]):
    """
    Read a file's contents with optional line range.

    Returns file content with line numbers (cat -n style).
    Default limit is 2000 lines; use offset+limit to page through large files.
    """

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            f"Read the contents of a file (up to {_DEFAULT_LIMIT} lines by default).\n\n"
            "IMPORTANT: Use this tool instead of running `cat`, `head`, or `tail` in bash.\n\n"
            "Returns content with line numbers in cat -n format for easy reference.\n\n"
            "Parameters:\n"
            "  file_path  — absolute path to the file (required)\n"
            f"  limit      — lines to read (default {_DEFAULT_LIMIT}; increase if needed)\n"
            "  offset     — 1-indexed line number to start from (for paging large files)\n\n"
            "Examples:\n"
            '  Read first 2000 lines:  {"file_path": "/abs/path/to/file.py"}\n'
            '  Read lines 100-200:     {"file_path": "/abs/path/to/file.py", "offset": 100, "limit": 100}\n'
            '  Read from line 500:     {"file_path": "/abs/path/to/file.py", "offset": 500}'
        )

    @property
    def input_schema(self) -> type[FileReadInput]:
        return FileReadInput

    async def execute(self, input_data: FileReadInput) -> ToolResult:
        try:
            path = Path(input_data.file_path).expanduser().resolve()

            if not path.exists():
                return ToolResult(
                    success=False, output="",
                    error=f"File not found: {input_data.file_path}",
                )

            if not path.is_file():
                return ToolResult(
                    success=False, output="",
                    error=f"Not a file: {input_data.file_path}",
                )

            # Refuse obvious binary files
            if path.suffix.lower() in _BINARY_EXTENSIONS:
                return ToolResult(
                    success=False, output="",
                    error=(
                        f"Binary file type ({path.suffix}) — "
                        "use bash or a dedicated tool for this format."
                    ),
                )

            # Size guard (10 MB)
            if path.stat().st_size > 10 * 1024 * 1024:
                return ToolResult(
                    success=False, output="",
                    error=(
                        "File exceeds 10 MB. Use offset+limit to read specific sections, "
                        "or grep to find the relevant part first."
                    ),
                )

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total = len(all_lines)

            # offset is 1-indexed (matches the line numbers in output)
            start_1 = max(1, input_data.offset or 1)
            start_0 = start_1 - 1  # convert to 0-indexed slice

            limit = input_data.limit if input_data.limit is not None else _DEFAULT_LIMIT
            end_0 = min(start_0 + limit, total)

            selected = all_lines[start_0:end_0]

            # cat -n style: "     N\t<content>"
            output_lines: list[str] = []
            for i, line in enumerate(selected, start=start_1):
                output_lines.append(f"{i:6d}\t{line.rstrip()}")

            output = "\n".join(output_lines)

            # Truncation notice
            if end_0 < total:
                output += (
                    f"\n\n"
                    f"[read_file] Showing lines {start_1}–{end_0} of {total}. "
                    f"Use offset={end_0 + 1} to continue reading."
                )
            else:
                output += f"\n\n[read_file] {total} lines total."

            return ToolResult(success=True, output=output)

        except UnicodeDecodeError:
            return ToolResult(
                success=False, output="",
                error="File contains non-UTF-8 content (binary?). Try bash if needed.",
            )
        except PermissionError:
            return ToolResult(
                success=False, output="",
                error=f"Permission denied: {input_data.file_path}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"read_file error: {e}")
