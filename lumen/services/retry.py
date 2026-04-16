"""
API retry handler — exponential backoff with jitter, 529 fallback, overflow shrink.

Mirrors src/services/api/withRetry.ts:
  - Exponential backoff: BASE_DELAY * 2^(attempt-1) + 25% jitter
  - Retry-After header honoring
  - 529 overload → fallback model after MAX_529_RETRIES
  - max_tokens overflow → auto-shrink and retry
  - Rate limit (429) with backoff
  - Configurable max retries
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from .errors import ClassifiedError, ErrorType, classify_error, parse_max_tokens_overflow

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Constants ────────────────────────────────────────────────────────────────

BASE_DELAY_MS = 500
MAX_DELAY_MS = 32_000
MAX_RETRIES = 10
MAX_529_RETRIES = 3
JITTER_FACTOR = 0.25


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = MAX_RETRIES
    base_delay_ms: int = BASE_DELAY_MS
    max_delay_ms: int = MAX_DELAY_MS
    max_529_retries: int = MAX_529_RETRIES
    fallback_model: str | None = None
    on_retry: Callable[[int, ClassifiedError, float], Any] | None = None
    on_fallback: Callable[[str], Any] | None = None
    on_token_adjust: Callable[[int, int], Any] | None = None


@dataclass
class RetryState:
    """Mutable state tracked across retry attempts."""
    attempt: int = 0
    overload_retries: int = 0
    current_max_tokens: int | None = None
    used_fallback: bool = False
    fallback_model: str | None = None


@dataclass
class RetryResult:
    """Result from a retried operation."""
    result: Any
    state: RetryState


# ── Backoff calculation ──────────────────────────────────────────────────────

def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Exponential backoff with jitter, in seconds."""
    delay_ms = config.base_delay_ms * (2 ** (attempt - 1))
    delay_ms = min(delay_ms, config.max_delay_ms)
    # Add 25% random jitter
    jitter = delay_ms * JITTER_FACTOR * random.random()
    return (delay_ms + jitter) / 1000.0


# ── Core retry logic ────────────────────────────────────────────────────────

async def with_retry(
    fn: Callable[..., Awaitable[T]],
    config: RetryConfig | None = None,
    max_tokens: int | None = None,
    **fn_kwargs: Any,
) -> RetryResult:
    """
    Execute an async function with retry logic.

    Handles:
      - 429 rate limit: backoff with Retry-After header
      - 529 server overload: retry up to MAX_529_RETRIES, then fallback
      - Connection/timeout errors: retry with backoff
      - max_tokens overflow: auto-shrink and retry
      - Auth/invalid errors: fail immediately

    Parameters
    ----------
    fn : async callable
        The function to call. Must accept **kwargs including 'max_tokens'
        and optionally 'model'.
    config : RetryConfig
        Retry configuration.
    max_tokens : int | None
        Current max_tokens value (for overflow adjustment).
    **fn_kwargs : Any
        Additional kwargs passed to fn on each attempt.

    Returns
    -------
    RetryResult with the successful result and retry state.

    Raises
    ------
    The last classified error's original exception if all retries exhausted.
    """
    cfg = config or RetryConfig()
    state = RetryState(current_max_tokens=max_tokens)

    last_error: ClassifiedError | None = None

    while state.attempt <= cfg.max_retries:
        state.attempt += 1

        try:
            # Apply current max_tokens if adjusted
            kwargs = dict(fn_kwargs)
            if state.current_max_tokens is not None:
                kwargs["max_tokens"] = state.current_max_tokens
            if state.fallback_model:
                kwargs["model"] = state.fallback_model

            result = await fn(**kwargs)
            return RetryResult(result=result, state=state)

        except Exception as exc:
            # Classify the error
            headers = {}
            resp = getattr(exc, "response", None)
            if resp is not None:
                headers = dict(getattr(resp, "headers", {}))

            classified = classify_error(exc, headers=headers)
            last_error = classified

            logger.warning(
                "API error (attempt %d/%d): %s [%s] status=%s",
                state.attempt, cfg.max_retries,
                classified.error_type.value,
                classified.message[:100],
                classified.status_code,
            )

            # ── Non-retryable: fail immediately ──────────────────────
            if not classified.retryable and not classified.should_adjust_tokens:
                raise

            # ── max_tokens overflow: shrink and retry ────────────────
            if classified.should_adjust_tokens and classified.max_tokens_info:
                info = classified.max_tokens_info
                new_max = info["available"]
                if new_max <= 0:
                    logger.error("No available tokens after overflow adjustment")
                    raise

                old_max = state.current_max_tokens or max_tokens or 0
                state.current_max_tokens = new_max
                logger.info(
                    "Adjusting max_tokens: %d → %d (available: %d)",
                    old_max, new_max, info["available"],
                )
                if cfg.on_token_adjust:
                    cfg.on_token_adjust(old_max, new_max)
                # Retry immediately without backoff
                continue

            # ── 529 overload: retry limited times, then fallback ─────
            if classified.error_type == ErrorType.SERVER_OVERLOAD:
                state.overload_retries += 1
                if state.overload_retries > cfg.max_529_retries:
                    if cfg.fallback_model and not state.used_fallback:
                        state.used_fallback = True
                        state.fallback_model = cfg.fallback_model
                        state.overload_retries = 0
                        logger.info(
                            "Server overloaded after %d retries, falling back to %s",
                            cfg.max_529_retries, cfg.fallback_model,
                        )
                        if cfg.on_fallback:
                            cfg.on_fallback(cfg.fallback_model)
                        continue
                    # No fallback available
                    raise

            # ── Calculate delay ──────────────────────────────────────
            if classified.retry_after:
                delay = classified.retry_after
            else:
                delay = _calculate_delay(state.attempt, cfg)

            logger.info(
                "Retrying in %.1fs (attempt %d/%d)...",
                delay, state.attempt, cfg.max_retries,
            )

            if cfg.on_retry:
                cfg.on_retry(state.attempt, classified, delay)

            await asyncio.sleep(delay)

    # All retries exhausted
    if last_error and last_error.original:
        raise last_error.original
    raise RuntimeError(f"All {cfg.max_retries} retries exhausted")
