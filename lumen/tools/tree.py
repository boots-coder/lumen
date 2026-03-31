"""
Tree tool — smart directory structure visualization.

Shows project layout with file sizes, excluding noise directories.
This is the FIRST tool to call when exploring an unfamiliar codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

# Directories that are almost never interesting for code reading
_EXCLUDE_DIRS = {
    ".git", ".svn", ".hg", ".bzr",
    "node_modules", ".pnp",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    ".venv", "venv", "env", ".env",
    "dist", "build", ".next", ".nuxt", ".output",
    "coverage", ".coverage",
    ".tox", ".nox",
    "vendor",
    ".idea", ".vscode",
    "*.egg-info",
}

# File extensions that are almost never worth reading
_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd",
    ".so", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".rar",
    ".lock",  # package-lock.json, poetry.lock etc — too noisy
}

_MAX_FILES_PER_DIR = 30   # collapse very large dirs
_MAX_TOTAL_LINES = 200    # keep output token-efficient


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.1f} MB"


def _build_tree(
    path: Path,
    prefix: str,
    depth: int,
    max_depth: int,
    lines: list[str],
    show_size: bool,
) -> None:
    if depth > max_depth:
        return
    if len(lines) >= _MAX_TOTAL_LINES:
        return

    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        lines.append(f"{prefix}[permission denied]")
        return

    # Filter
    dirs = [e for e in entries if e.is_dir() and e.name not in _EXCLUDE_DIRS
            and not e.name.endswith(".egg-info")]
    files = [e for e in entries if e.is_file()
             and e.suffix.lower() not in _SKIP_EXTENSIONS]

    children = dirs + files
    for i, entry in enumerate(children):
        if len(lines) >= _MAX_TOTAL_LINES:
            remaining = len(children) - i
            lines.append(f"{prefix}... ({remaining} more)")
            break

        connector = "└── " if i == len(children) - 1 else "├── "
        extension = "    " if i == len(children) - 1 else "│   "

        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            _build_tree(
                entry, prefix + extension, depth + 1, max_depth,
                lines, show_size,
            )
        else:
            size_str = ""
            if show_size:
                try:
                    size_str = f"  ({_fmt_size(entry.stat().st_size)})"
                except OSError:
                    pass
            lines.append(f"{prefix}{connector}{entry.name}{size_str}")


class TreeInput(BaseModel):
    path: str = Field(
        default=".",
        description="Directory to show (default: current working directory)",
    )
    depth: int = Field(
        default=3,
        description="How many levels deep to show (default 3, max 6)",
    )
    show_size: bool = Field(
        default=True,
        description="Show file sizes next to file names",
    )


class TreeTool(Tool[TreeInput]):
    """
    Show the directory structure of a project.

    Use this as the FIRST step when exploring a codebase — it gives you
    a map of the project before you start reading individual files.

    Noise directories (node_modules, __pycache__, .git, venv, dist, etc.)
    are automatically excluded.
    """

    @property
    def name(self) -> str:
        return "tree"

    @property
    def description(self) -> str:
        return (
            "Show the directory structure of a project (smart tree with noise excluded).\n\n"
            "Use this FIRST when exploring an unfamiliar codebase to understand the layout.\n"
            "Automatically excludes: node_modules, __pycache__, .git, venv, dist, build, etc.\n\n"
            "Parameters:\n"
            "  path       — directory to show (default: current working directory)\n"
            "  depth      — levels deep (default 3, use 2 for large projects)\n"
            "  show_size  — show file sizes (default true)\n\n"
            "Examples:\n"
            '  Project overview:    {"path": "."}\n'
            '  Shallow overview:    {"depth": 2}\n'
            '  Deep dive into src:  {"path": "/abs/path/src", "depth": 4}'
        )

    @property
    def input_schema(self) -> type[TreeInput]:
        return TreeInput

    async def execute(self, input_data: TreeInput) -> ToolResult:
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

            max_depth = min(max(1, input_data.depth), 6)
            lines: list[str] = [f"{base.name}/"]
            _build_tree(base, "", 1, max_depth, lines, input_data.show_size)

            if len(lines) >= _MAX_TOTAL_LINES:
                lines.append(
                    f"\n[tree] Output truncated at {_MAX_TOTAL_LINES} lines. "
                    "Use a smaller depth or target a specific subdirectory."
                )

            return ToolResult(success=True, output="\n".join(lines))

        except Exception as e:
            return ToolResult(success=False, output="", error=f"tree error: {e}")
