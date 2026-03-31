"""
Glob tool — file pattern matching sorted by modification time.

Design:
- Results sorted by mtime descending (recently modified files first)
- Hard limit of 100 results (returns truncated=true if exceeded)
- Reports duration for diagnostics
- Relative paths to save tokens
"""

from __future__ import annotations

import glob as glob_module
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

_MAX_RESULTS = 100


class GlobInput(BaseModel):
    """Input schema for Glob tool."""

    pattern: str = Field(
        description=(
            "Glob pattern to match files. Examples:\n"
            "  '**/*.py'        — all Python files recursively\n"
            "  'src/**/*.ts'    — TypeScript files under src/\n"
            "  'test_*.py'      — files starting with test_\n"
            "  '*.{js,ts}'      — JavaScript or TypeScript in current dir"
        )
    )
    path: str = Field(
        default=".",
        description="Base directory to search in (default: current directory)",
    )


class GlobTool(Tool[GlobInput]):
    """
    Find files by name pattern, sorted by modification time (newest first).

    Use this to discover files before reading them.
    Results are capped at 100; use a more specific pattern if truncated.
    """

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern (sorted by modification time, newest first).\n\n"
            "IMPORTANT: Use this tool instead of running `find` or `ls` in bash.\n\n"
            "Returns up to 100 results. If truncated, refine your pattern.\n\n"
            "Pattern syntax:\n"
            "  *   — matches anything except /\n"
            "  **  — matches any number of directories\n"
            "  ?   — matches a single character\n"
            "  {}  — brace expansion, e.g. *.{py,ts}\n\n"
            "Examples:\n"
            '  All Python files:     {"pattern": "**/*.py"}\n'
            '  TypeScript in src/:   {"pattern": "src/**/*.ts"}\n'
            '  Test files:           {"pattern": "test_*.py", "path": "/abs/path"}'
        )

    @property
    def input_schema(self) -> type[GlobInput]:
        return GlobInput

    async def execute(self, input_data: GlobInput) -> ToolResult:
        try:
            base = Path(input_data.path).expanduser().resolve()

            if not base.exists():
                return ToolResult(
                    success=False, output="",
                    error=f"Path not found: {input_data.path}",
                )

            if not base.is_dir():
                return ToolResult(
                    success=False, output="",
                    error=f"Not a directory: {input_data.path}",
                )

            t0 = time.monotonic()
            full_pattern = str(base / input_data.pattern)
            matches = glob_module.glob(full_pattern, recursive=True)
            duration_ms = int((time.monotonic() - t0) * 1000)

            # Files only
            matches = [m for m in matches if Path(m).is_file()]

            # Sort by mtime descending (recently modified first)
            def _mtime(p: str) -> float:
                try:
                    return Path(p).stat().st_mtime
                except OSError:
                    return 0.0

            matches.sort(key=_mtime, reverse=True)

            # Make relative to base (saves tokens)
            rel_matches: list[str] = []
            for m in matches:
                try:
                    rel_matches.append(str(Path(m).relative_to(base)))
                except ValueError:
                    rel_matches.append(m)

            total = len(rel_matches)
            truncated = total > _MAX_RESULTS
            shown = rel_matches[:_MAX_RESULTS]

            if not shown:
                output = f"[glob] No files found matching '{input_data.pattern}' in {base}"
            else:
                header = (
                    f"[glob] {total} file(s) matched '{input_data.pattern}'"
                    + (" (showing first 100 — refine pattern for more)" if truncated else "")
                    + f"  [{duration_ms} ms]"
                )
                output = header + "\n\n" + "\n".join(shown)

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output="", error=f"glob error: {e}")
