"""
Auto-compaction decision logic.

Auto-compaction decision logic.

Key constants (all token counts):
  AUTOCOMPACT_BUFFER_TOKENS  = 13_000   — trigger headroom before hard limit
  MAX_OUTPUT_RESERVE         = 20_000   — tokens reserved for the summary output
  MAX_CONSECUTIVE_FAILURES   = 3        — circuit-breaker threshold

Threshold formula:
  effective_window = context_window - min(max_output_tokens, MAX_OUTPUT_RESERVE)
  autocompact_threshold = effective_window - AUTOCOMPACT_BUFFER_TOKENS

  For a 200K model with 32K output:
    effective_window      = 200_000 - 20_000 = 180_000
    autocompact_threshold = 180_000 - 13_000 = 167_000

  For a 128K model with 16K output:
    effective_window      = 128_000 - 16_000 = 112_000
    autocompact_threshold = 112_000 - 13_000 = 99_000
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..tokens.counter import get_context_window, get_max_output_tokens

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────
AUTOCOMPACT_BUFFER_TOKENS: int = 13_000
MAX_OUTPUT_RESERVE: int = 20_000
MAX_CONSECUTIVE_FAILURES: int = 3

# Warning / error display thresholds (shown to user before compaction triggers)
WARNING_THRESHOLD_BUFFER: int = 20_000
ERROR_THRESHOLD_BUFFER: int = 20_000

# /compact (manual) blocking threshold — stop accepting input if this close to limit
MANUAL_COMPACT_BUFFER: int = 3_000


@dataclass(frozen=True)
class ContextWindowState:
    """
    Snapshot of where we are in the context window.

    All values in tokens.
    """
    total_tokens: int
    context_window: int
    effective_window: int
    autocompact_threshold: int
    warning_threshold: int
    blocking_threshold: int

    @property
    def percent_used(self) -> float:
        if self.context_window == 0:
            return 0.0
        return round(self.total_tokens / self.context_window * 100, 1)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.context_window - self.total_tokens)

    @property
    def should_warn(self) -> bool:
        return self.total_tokens >= self.warning_threshold

    @property
    def should_compact(self) -> bool:
        return self.total_tokens >= self.autocompact_threshold

    @property
    def is_blocked(self) -> bool:
        """True when the context is so full we must compact before accepting input."""
        return self.total_tokens >= self.blocking_threshold


def calculate_thresholds(
    model: str,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Return (effective_window, autocompact_threshold, warning_threshold, blocking_threshold).

    Users can override context_window / max_output_tokens to suit models not
    in our registry.
    """
    cw = context_window or get_context_window(model)
    mo = max_output_tokens or get_max_output_tokens(model)

    reserved = min(mo, MAX_OUTPUT_RESERVE)
    effective = cw - reserved
    autocompact = effective - AUTOCOMPACT_BUFFER_TOKENS
    warning = effective - WARNING_THRESHOLD_BUFFER
    blocking = effective - MANUAL_COMPACT_BUFFER

    return effective, autocompact, warning, blocking


def assess_context_window(
    total_tokens: int,
    model: str,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
) -> ContextWindowState:
    """
    Compute the full ContextWindowState for the current token usage.

    This is the main decision function — call it after every API response
    to determine whether to compact.
    """
    cw = context_window or get_context_window(model)
    effective, autocompact, warning, blocking = calculate_thresholds(
        model, context_window, max_output_tokens
    )

    state = ContextWindowState(
        total_tokens=total_tokens,
        context_window=cw,
        effective_window=effective,
        autocompact_threshold=autocompact,
        warning_threshold=warning,
        blocking_threshold=blocking,
    )

    if state.is_blocked:
        logger.warning(
            "Context window critically full: %d/%d tokens (%.1f%%). "
            "Compaction required before next message.",
            total_tokens, cw, state.percent_used,
        )
    elif state.should_compact:
        logger.info(
            "Auto-compact threshold reached: %d/%d tokens (%.1f%%).",
            total_tokens, cw, state.percent_used,
        )
    elif state.should_warn:
        logger.info(
            "Context window %.1f%% full (%d tokens remaining).",
            state.percent_used, state.tokens_remaining,
        )

    return state
