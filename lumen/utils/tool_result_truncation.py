"""
Tool result truncation — prevents oversized tool output from blowing up context.

Mirrors src/utils/toolResultStorage.ts:
  - Per-tool result size limit (default 50K chars)
  - Aggregate per-message budget (200K chars)
  - Large results: truncate with preview + indicator
  - Empty results: inject "(completed with no output)" marker
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MAX_RESULT_CHARS = 50_000          # Per-tool default
MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = 200_000  # Aggregate per message
PREVIEW_BYTES = 2000                       # First N bytes as preview
TRUNCATION_MARKER = "\n\n... [truncated — {removed} chars removed, {kept} chars kept]"
EMPTY_MARKER = "({tool_name} completed with no output)"


# ── Per-tool limits ──────────────────────────────────────────────────────────

_TOOL_LIMITS: dict[str, int] = {
    "bash": 100_000,         # Bash can produce large output
    "read_file": 80_000,     # File reads can be large
    "grep": 60_000,          # Grep results can be long
    "tree": 30_000,          # Tree is usually shorter
    "glob": 30_000,
    "definitions": 40_000,
}


def get_tool_limit(tool_name: str) -> int:
    """Get the max result chars for a tool."""
    return _TOOL_LIMITS.get(tool_name, DEFAULT_MAX_RESULT_CHARS)


# ── Truncation ───────────────────────────────────────────────────────────────

def truncate_tool_result(
    content: str,
    tool_name: str,
    max_chars: int | None = None,
) -> str:
    """
    Truncate a tool result if it exceeds the limit.

    Returns the original content if within limit, or a truncated
    version with a preview and truncation marker.
    """
    if not content:
        return EMPTY_MARKER.format(tool_name=tool_name)

    limit = max_chars or get_tool_limit(tool_name)

    if len(content) <= limit:
        return content

    # Truncate: keep first PREVIEW_BYTES, add marker
    preview_end = min(PREVIEW_BYTES, limit)

    # Try to truncate at a newline boundary
    newline_pos = content.rfind("\n", 0, preview_end)
    if newline_pos > preview_end // 2:
        preview_end = newline_pos + 1

    preview = content[:preview_end]
    removed = len(content) - preview_end
    marker = TRUNCATION_MARKER.format(removed=removed, kept=preview_end)

    logger.debug(
        "Truncated %s result: %d → %d chars",
        tool_name, len(content), preview_end,
    )

    return preview + marker


def truncate_tool_results_batch(
    results: list[tuple[str, str]],  # [(tool_name, content), ...]
    max_total_chars: int = MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
) -> list[str]:
    """
    Truncate a batch of tool results to fit within aggregate budget.

    First applies per-tool limits, then enforces total budget
    by proportionally reducing oversized results.
    """
    # Step 1: Apply per-tool limits
    truncated = []
    for tool_name, content in results:
        truncated.append(truncate_tool_result(content, tool_name))

    # Step 2: Check aggregate
    total = sum(len(r) for r in truncated)
    if total <= max_total_chars:
        return truncated

    # Step 3: Proportionally reduce
    ratio = max_total_chars / total
    final = []
    for i, (tool_name, _) in enumerate(results):
        allowed = int(len(truncated[i]) * ratio)
        if len(truncated[i]) > allowed:
            final.append(truncate_tool_result(truncated[i], tool_name, max_chars=allowed))
        else:
            final.append(truncated[i])

    return final
