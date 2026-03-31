"""
Bash tool — shell command execution.

Design philosophy:
- Actively discourages using bash as a substitute for dedicated tools
- Default timeout 120 s
- stderr shown separately so model can distinguish output from errors
"""

from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from .base import Tool, ToolResult


class BashInput(BaseModel):
    """Input schema for Bash tool."""

    command: str = Field(description="Shell command to execute")
    timeout: int = Field(
        120,
        description="Timeout in seconds (default 120, max 600)",
    )


class BashTool(Tool[BashInput]):
    """
    Execute a bash shell command.

    Use ONLY when no dedicated tool applies.
    For common operations prefer the dedicated tools — they are faster and
    more token-efficient than bash equivalents.
    """

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a bash shell command.\n\n"
            "IMPORTANT: Avoid using bash for these common tasks — use the dedicated tool instead:\n"
            "  cat / head / tail  →  use read_file\n"
            "  find / ls          →  use glob\n"
            "  grep / rg          →  use grep\n\n"
            "Good uses of bash:\n"
            "  - git operations:  git log, git diff, git status\n"
            "  - Build / test:    pytest, make, npm test\n"
            "  - System info:     pwd, whoami, df, ps\n"
            "  - Package info:    pip show, pip list\n"
            "  - Directory tree:  tree -L 2\n\n"
            "Parameters:\n"
            "  command  — shell command (required)\n"
            "  timeout  — seconds before killing (default 120)\n\n"
            "Examples:\n"
            '  {"command": "git log --oneline -10"}\n'
            '  {"command": "pytest tests/ -x -q"}\n'
            '  {"command": "tree -L 2 /path/to/project"}'
        )

    @property
    def input_schema(self) -> type[BashInput]:
        return BashInput

    async def execute(self, input_data: BashInput) -> ToolResult:
        try:
            timeout = min(max(1, input_data.timeout), 600)

            result = subprocess.run(
                input_data.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout.rstrip()
            stderr = result.stderr.rstrip()

            # Build output: stdout first, then stderr labelled separately
            parts: list[str] = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr]\n{stderr}")
            if not parts:
                parts.append("(no output)")

            output = "\n\n".join(parts)

            if result.returncode != 0:
                # Return as failure but still include all output so the model
                # can reason about what went wrong.
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Exit code {result.returncode}",
                )

            return ToolResult(success=True, output=output)

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, output="",
                error=f"Command timed out after {input_data.timeout} s",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"bash error: {e}")
