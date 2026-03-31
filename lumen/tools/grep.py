"""
Grep tool — ripgrep-powered file content search.

Design:
- Default output_mode is "files_with_matches" (token-efficient; returns only paths)
- Switch to "content" for actual matching lines with context
- VCS directories (.git / .svn / .hg) excluded automatically
- Results sorted by modification time (newest first) in files_with_matches mode
- --max-columns 500 prevents minified-code noise
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

# Search for ripgrep in common locations; fall back to plain grep
_RG = (
    shutil.which("rg")
    or shutil.which("rg", path="/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/opt/homebrew/bin")
    or "rg"
)

_VCS_EXCLUDES = [".git", ".svn", ".hg", ".bzr", ".jj"]


class GrepInput(BaseModel):
    """Input schema for Grep tool."""

    pattern: str = Field(description="Regular expression pattern to search for (ripgrep syntax)")
    path: str = Field(
        default=".",
        description="Directory or file to search (default: current working directory)",
    )
    glob: str | None = Field(
        None,
        description=(
            "File-name glob filter, e.g. '*.py' or '*.{ts,tsx}'. "
            "Passed directly to ripgrep --glob."
        ),
    )
    output_mode: str = Field(
        default="files_with_matches",
        description=(
            "Output mode:\n"
            "  'files_with_matches' — return only file paths (default, most token-efficient)\n"
            "  'content'            — return matching lines with optional context\n"
            "  'count'              — return match counts per file"
        ),
    )
    case_insensitive: bool = Field(False, description="Ignore case (-i flag)")
    context: int | None = Field(
        None,
        description="Lines of context to show before and after each match (content mode only)",
    )
    head_limit: int = Field(
        250,
        description="Truncate output to this many lines/entries. Pass 0 for unlimited.",
    )
    multiline: bool = Field(
        False,
        description="Enable multiline mode so patterns can span lines (-U --multiline-dotall)",
    )


class GrepTool(Tool[GrepInput]):
    """
    Fast file content search using ripgrep.

    Prefer output_mode='files_with_matches' (default) to get a cheap file list,
    then use read_file to read the specific files that are relevant.
    Use output_mode='content' when you need the actual matching lines.
    """

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents with ripgrep (full regex support).\n\n"
            "IMPORTANT: Use this tool instead of running `grep` or `rg` in bash.\n\n"
            "output_mode options:\n"
            "  'files_with_matches' (default) — returns only file paths. Most token-efficient.\n"
            "    Use this first to find which files match, then read_file those files.\n"
            "  'content' — returns matching lines. Add context=N for N lines around each match.\n"
            "  'count'   — returns match count per file.\n\n"
            "The glob parameter filters by filename pattern, e.g. '*.py' or '*.{ts,tsx}'.\n"
            "VCS directories (.git, .svn, .hg) are always excluded.\n\n"
            "Examples:\n"
            '  Find files containing "Agent": {"pattern": "class Agent", "glob": "*.py"}\n'
            '  See matching lines: {"pattern": "def chat", "output_mode": "content", "context": 3}\n'
            '  Count per file: {"pattern": "TODO", "output_mode": "count"}'
        )

    @property
    def input_schema(self) -> type[GrepInput]:
        return GrepInput

    async def execute(self, input_data: GrepInput) -> ToolResult:
        try:
            search_path = Path(input_data.path).expanduser().resolve()

            if not search_path.exists():
                return ToolResult(
                    success=False, output="",
                    error=f"Path not found: {input_data.path}",
                )

            cmd: list[str] = [_RG]

            # ── Output mode ───────────────────────────────────────────────────
            mode = input_data.output_mode.lower()
            if mode == "files_with_matches":
                cmd.append("-l")
            elif mode == "count":
                cmd.append("-c")
            else:
                # content mode: show line numbers by default
                cmd.append("-n")

            # ── Flags ─────────────────────────────────────────────────────────
            if input_data.case_insensitive:
                cmd.append("-i")

            if input_data.multiline:
                cmd.extend(["-U", "--multiline-dotall"])

            if input_data.context is not None and mode == "content":
                cmd.extend(["-C", str(input_data.context)])

            # Prevent minified / base64 lines from cluttering results
            cmd.extend(["--max-columns", "500"])

            # Exclude VCS directories
            for vcs in _VCS_EXCLUDES:
                cmd.extend(["--glob", f"!{vcs}"])

            # Hidden files (like .env, .github)
            cmd.append("--hidden")

            # File-name filter
            if input_data.glob:
                cmd.extend(["--glob", input_data.glob])

            # Pattern — use -e to handle patterns starting with "-"
            cmd.extend(["-e", input_data.pattern])

            # Search path
            cmd.append(str(search_path))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # rc=1 means no matches (not an error in ripgrep)
            if result.returncode == 1:
                return ToolResult(
                    success=True,
                    output=f"No matches found for: {input_data.pattern}",
                )

            if result.returncode not in (0, 1):
                err = result.stderr.strip() or "ripgrep failed"
                return ToolResult(success=False, output="", error=err)

            raw = result.stdout.strip()
            if not raw:
                return ToolResult(
                    success=True,
                    output=f"No matches found for: {input_data.pattern}",
                )

            lines = raw.splitlines()

            # ── Sort files_with_matches by mtime (newest first) ───────────────
            if mode == "files_with_matches":
                def _mtime(p: str) -> float:
                    try:
                        return Path(p).stat().st_mtime
                    except OSError:
                        return 0.0

                lines = sorted(lines, key=_mtime, reverse=True)

            # ── Convert absolute paths → relative for token efficiency ─────────
            try:
                lines = [
                    str(Path(ln).relative_to(search_path))
                    if Path(ln).is_absolute()
                    else ln
                    for ln in lines
                ]
            except ValueError:
                pass  # keep absolute if relative conversion fails

            # ── head_limit ────────────────────────────────────────────────────
            limit = input_data.head_limit
            truncated = False
            if limit and len(lines) > limit:
                lines = lines[:limit]
                truncated = True

            # ── Build output ──────────────────────────────────────────────────
            header = (
                f"[grep] pattern={input_data.pattern!r}  "
                f"mode={mode}  "
                f"results={len(lines)}"
                + ("  (truncated)" if truncated else "")
            )
            output = header + "\n\n" + "\n".join(lines)

            return ToolResult(success=True, output=output)

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, output="",
                error="ripgrep timed out (>30 s)",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False, output="",
                error="ripgrep (rg) not found — install it with: brew install ripgrep",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"grep error: {e}")
