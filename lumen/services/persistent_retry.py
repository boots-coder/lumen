"""
Persistent Retry — unattended retry with escalation for CI/batch scenarios.

Unlike the standard retry (which gives up after max_retries), persistent retry:
  - Retries indefinitely on transient errors (with increasing backoff)
  - Escalates to fallback models after N failures
  - Logs all attempts for post-mortem analysis
  - Can notify via webhook on repeated failures
  - Has a total timeout (e.g., 1 hour) as ultimate circuit breaker
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class PersistentRetryConfig:
    """Configuration for persistent retry behavior."""
    enabled: bool = False
    total_timeout_seconds: int = 3600  # 1 hour
    escalation_threshold: int = 5     # Switch to fallback after N failures
    fallback_models: list[str] = field(default_factory=list)
    webhook_url: str | None = None    # Notify on failure
    log_file: Path | None = None      # Log all attempts
    backoff_max_seconds: int = 300    # 5 minutes max backoff
    notify_after_failures: int = 10   # POST to webhook after N failures


@dataclass
class RetryLogEntry:
    """A single retry attempt log entry."""
    timestamp: str
    attempt: int
    model: str
    error: str
    latency_ms: float


@dataclass
class PersistentRetryResult:
    """Result from a persistent retry operation."""
    result: Any
    total_attempts: int
    models_tried: list[str]
    total_time_ms: float
    log_entries: list[RetryLogEntry] = field(default_factory=list)


# ── Manager ──────────────────────────────────────────────────────────────────

class PersistentRetryManager:
    """
    Manages persistent retry with escalation and notification.

    Retry loop:
      1. Call fn() with current model
      2. On failure: log, backoff, and retry
      3. After escalation_threshold failures: switch to next fallback model
      4. After notify_after_failures: POST to webhook_url
      5. Stop on success, total_timeout exceeded, or all models exhausted

    The backoff follows exponential growth with jitter, capped at
    backoff_max_seconds.
    """

    BASE_DELAY_SECONDS = 1.0
    JITTER_FACTOR = 0.25

    async def execute_with_persistent_retry(
        self,
        fn: Callable[..., Awaitable[T]],
        config: PersistentRetryConfig,
        *,
        initial_model: str = "",
        **fn_kwargs: Any,
    ) -> PersistentRetryResult:
        """
        Execute fn with persistent retry until success or circuit break.

        Parameters
        ----------
        fn : async callable
            The function to retry. Must accept **kwargs including 'model'.
        config : PersistentRetryConfig
            Retry configuration.
        initial_model : str
            The initial model to use.
        **fn_kwargs : Any
            Additional kwargs passed to fn on each attempt.

        Returns
        -------
        PersistentRetryResult with the successful result and metadata.

        Raises
        ------
        TimeoutError
            When total_timeout_seconds is exceeded.
        RuntimeError
            When all models are exhausted without success.
        """
        start_time = time.monotonic()
        attempt = 0
        log_entries: list[RetryLogEntry] = []
        models_tried: list[str] = []
        notified = False

        # Build model escalation chain
        current_model = initial_model
        if current_model and current_model not in models_tried:
            models_tried.append(current_model)

        # Track consecutive failures for escalation
        consecutive_failures = 0
        fallback_index = 0

        while True:
            attempt += 1
            elapsed = time.monotonic() - start_time

            # Circuit breaker: total timeout
            if elapsed > config.total_timeout_seconds:
                logger.error(
                    "Persistent retry timed out after %d attempts (%.1fs)",
                    attempt - 1, elapsed,
                )
                self._write_log(config, log_entries)
                raise TimeoutError(
                    f"Persistent retry timed out after {attempt - 1} attempts "
                    f"({elapsed:.1f}s > {config.total_timeout_seconds}s limit)"
                )

            # Attempt the call
            attempt_start = time.monotonic()
            try:
                kwargs = dict(fn_kwargs)
                if current_model:
                    kwargs["model"] = current_model

                result = await fn(**kwargs)

                # Success
                total_time_ms = (time.monotonic() - start_time) * 1000
                self._write_log(config, log_entries)

                return PersistentRetryResult(
                    result=result,
                    total_attempts=attempt,
                    models_tried=list(models_tried),
                    total_time_ms=total_time_ms,
                    log_entries=log_entries,
                )

            except Exception as exc:
                latency_ms = (time.monotonic() - attempt_start) * 1000
                error_msg = str(exc)[:500]

                entry = RetryLogEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    attempt=attempt,
                    model=current_model,
                    error=error_msg,
                    latency_ms=round(latency_ms, 1),
                )
                log_entries.append(entry)

                logger.warning(
                    "Persistent retry attempt %d failed (model=%s): %s",
                    attempt, current_model, error_msg[:100],
                )

                consecutive_failures += 1

                # Escalation: switch to fallback model
                if (
                    consecutive_failures >= config.escalation_threshold
                    and fallback_index < len(config.fallback_models)
                ):
                    new_model = config.fallback_models[fallback_index]
                    fallback_index += 1
                    consecutive_failures = 0

                    logger.info(
                        "Escalating to fallback model: %s (after %d failures)",
                        new_model, config.escalation_threshold,
                    )
                    current_model = new_model
                    if current_model not in models_tried:
                        models_tried.append(current_model)

                # Check if all models exhausted
                if (
                    consecutive_failures >= config.escalation_threshold
                    and fallback_index >= len(config.fallback_models)
                ):
                    self._write_log(config, log_entries)
                    raise RuntimeError(
                        f"All models exhausted after {attempt} attempts. "
                        f"Models tried: {models_tried}"
                    ) from exc

                # Webhook notification
                if (
                    attempt >= config.notify_after_failures
                    and not notified
                    and config.webhook_url
                ):
                    notified = True
                    await self._notify_webhook(config, attempt, log_entries)

                # Backoff
                delay = self._calculate_backoff(attempt, config)
                logger.debug("Backing off %.1fs before attempt %d", delay, attempt + 1)
                await asyncio.sleep(delay)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _calculate_backoff(self, attempt: int, config: PersistentRetryConfig) -> float:
        """Exponential backoff with jitter, capped at backoff_max_seconds."""
        delay = self.BASE_DELAY_SECONDS * (2 ** (attempt - 1))
        delay = min(delay, config.backoff_max_seconds)
        jitter = delay * self.JITTER_FACTOR * random.random()
        return delay + jitter

    @staticmethod
    def _write_log(
        config: PersistentRetryConfig,
        entries: list[RetryLogEntry],
    ) -> None:
        """Write log entries to file if configured."""
        if not config.log_file or not entries:
            return

        try:
            import json
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            records = [
                {
                    "timestamp": e.timestamp,
                    "attempt": e.attempt,
                    "model": e.model,
                    "error": e.error,
                    "latency_ms": e.latency_ms,
                }
                for e in entries
            ]

            with log_path.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            logger.debug("Wrote %d retry log entries to %s", len(entries), log_path)
        except Exception as e:
            logger.error("Failed to write retry log: %s", e)

    @staticmethod
    async def _notify_webhook(
        config: PersistentRetryConfig,
        attempt: int,
        entries: list[RetryLogEntry],
    ) -> None:
        """POST failure notification to webhook URL."""
        if not config.webhook_url:
            return

        try:
            import urllib.request
            import json

            payload = json.dumps({
                "event": "persistent_retry_failures",
                "total_attempts": attempt,
                "threshold": config.notify_after_failures,
                "recent_errors": [
                    {"attempt": e.attempt, "model": e.model, "error": e.error}
                    for e in entries[-5:]
                ],
            }).encode("utf-8")

            req = urllib.request.Request(
                config.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=10),
            )

            logger.info(
                "Sent failure notification to webhook after %d attempts",
                attempt,
            )
        except Exception as e:
            logger.error("Failed to notify webhook %s: %s", config.webhook_url, e)
