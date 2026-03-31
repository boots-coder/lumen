"""
Git state capture — snapshot at session start.

Captured once at session start and cached (snapshot semantics).
Runs all git commands in parallel for speed.

Captured:
  - Current branch
  - Default/main branch (for PR context)
  - Working tree status (--short, truncated to 2000 chars)
  - Last 5 commits (--oneline)
  - Git user name
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_STATUS_CHARS = 2_000


async def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout. Returns '' on error."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "--no-optional-locks",  # avoid lock contention
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(cwd),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        return stdout.decode("utf-8", errors="replace").strip()
    except (OSError, asyncio.TimeoutError, FileNotFoundError):
        return ""


async def _is_git_repo(cwd: Path) -> bool:
    result = await _run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return result == "true"


async def _get_default_branch(cwd: Path) -> str:
    """Try to find the main/master branch name."""
    # Try remote HEAD
    remote = await _run_git(
        ["rev-parse", "--abbrev-ref", "origin/HEAD"], cwd
    )
    if remote.startswith("origin/"):
        return remote[len("origin/"):]

    # Fall back to local branch existence check
    for candidate in ("main", "master", "develop", "trunk"):
        result = await _run_git(["branch", "--list", candidate], cwd)
        if result:
            return candidate

    return "main"


async def get_git_state(cwd: Optional[Path] = None) -> Optional[str]:
    """
    Return a formatted git state string for the given directory.

    Returns None if:
      - Not a git repository
      - git is not installed
      - Any unexpected error occurs

    The returned string is injected into the system prompt once at session
    start and never refreshed (snapshot semantics).
    """
    cwd = (cwd or Path.cwd()).resolve()

    if not await _is_git_repo(cwd):
        return None

    # Fetch all data in parallel
    branch, status, log, user_name, default_branch = await asyncio.gather(
        _run_git(["branch", "--show-current"], cwd),
        _run_git(["status", "--short"], cwd),
        _run_git(["log", "--oneline", "-n", "5"], cwd),
        _run_git(["config", "user.name"], cwd),
        _get_default_branch(cwd),
    )

    # Truncate status if it exceeds the character limit
    status_note = ""
    if len(status) > MAX_STATUS_CHARS:
        status = status[:MAX_STATUS_CHARS]
        # Trim to last newline
        last_nl = status.rfind("\n")
        if last_nl > 0:
            status = status[:last_nl]
        status_note = (
            '\n... (truncated because it exceeds 2k characters. '
            'If you need more information, run "git status" manually)'
        )

    lines: list[str] = [
        "This is the git status at the start of the conversation. "
        "Note that this status is a snapshot in time and will not update during the conversation.",
        f"Current branch: {branch or '(unknown)'}",
        f"Main branch (you will usually use this for PRs): {default_branch}",
    ]
    if user_name:
        lines.append(f"Git user: {user_name}")

    lines.append("Status:")
    lines.append(status or "(clean)" + status_note)

    if log:
        lines.append("Recent commits:")
        lines.append(log)

    return "\n".join(lines)
