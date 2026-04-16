"""
API error classification — 12 error types with distinct handling strategies.

Mirrors src/services/api/errors.ts:
  Each error type has a recommended action (retry, fallback, abort, etc.)
  so the retry handler can make informed decisions.
"""

from __future__ import annotations

import re
from enum import Enum
from dataclasses import dataclass
from typing import Any


class ErrorType(str, Enum):
    """All classifiable API error types."""
    # ── Retryable ─────────────────────────────────────────
    RATE_LIMIT = "rate_limit"              # 429 — back off and retry
    SERVER_OVERLOAD = "server_overload"     # 529 — server busy, may fallback
    CONNECTION_ERROR = "connection_error"   # ECONNRESET / EPIPE / timeout
    API_TIMEOUT = "api_timeout"            # Read timeout

    # ── Adjustable ────────────────────────────────────────
    PROMPT_TOO_LONG = "prompt_too_long"    # Input exceeds context window
    MAX_TOKENS_OVERFLOW = "max_tokens_overflow"  # input + max_tokens > context

    # ── Non-retryable ─────────────────────────────────────
    AUTH_ERROR = "auth_error"              # 401 / 403 — bad key or expired
    INVALID_MODEL = "invalid_model"        # Model doesn't exist
    INVALID_REQUEST = "invalid_request"    # 400 — malformed payload
    REFUSAL = "refusal"                    # Model refused (safety)
    ABORTED = "aborted"                    # User or system cancelled

    # ── Catch-all ─────────────────────────────────────────
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    """An API error with classification and metadata."""
    error_type: ErrorType
    status_code: int | None
    message: str
    retryable: bool
    retry_after: float | None = None       # seconds, from Retry-After header
    max_tokens_info: dict | None = None    # parsed overflow details
    original: Exception | None = None

    @property
    def should_fallback(self) -> bool:
        return self.error_type == ErrorType.SERVER_OVERLOAD

    @property
    def should_adjust_tokens(self) -> bool:
        return self.error_type in (
            ErrorType.MAX_TOKENS_OVERFLOW,
            ErrorType.PROMPT_TOO_LONG,
        )


# ── Overflow parser ──────────────────────────────────────────────────────────

_OVERFLOW_RE = re.compile(
    r"(\d+)\s*\+\s*(\d+)\s*>\s*(\d+)"
)

def parse_max_tokens_overflow(message: str) -> dict | None:
    """
    Parse error like:
      'input length and `max_tokens` exceed context limit: 188059 + 20000 > 200000'
    Returns {input_tokens, max_tokens, context_limit, available}.
    """
    m = _OVERFLOW_RE.search(message)
    if not m:
        return None

    input_tokens = int(m.group(1))
    max_tokens = int(m.group(2))
    context_limit = int(m.group(3))
    safety = 1000
    available = max(0, context_limit - input_tokens - safety)

    return {
        "input_tokens": input_tokens,
        "max_tokens": max_tokens,
        "context_limit": context_limit,
        "available": available,
    }


# ── Classifier ───────────────────────────────────────────────────────────────

_CONNECTION_KEYWORDS = (
    "econnreset", "epipe", "econnrefused", "enotfound",
    "connection reset", "broken pipe", "connection refused",
    "network", "dns", "ssl",
)

_TIMEOUT_KEYWORDS = (
    "timeout", "timed out", "read timeout", "connect timeout",
)


def classify_error(
    error: Exception,
    status_code: int | None = None,
    headers: dict[str, str] | None = None,
) -> ClassifiedError:
    """
    Classify any exception into one of 12 error types.

    Works with httpx errors, generic exceptions, and anything with
    a status_code / message.
    """
    msg = str(error).lower()
    headers = headers or {}

    # Extract status code from httpx responses
    if status_code is None:
        sc = getattr(error, "status_code", None)
        if sc is None:
            resp = getattr(error, "response", None)
            if resp is not None:
                sc = getattr(resp, "status_code", None)
        status_code = sc

    # ── By status code ────────────────────────────────────────────────────

    if status_code == 429:
        retry_after = _parse_retry_after(headers.get("retry-after"))
        return ClassifiedError(
            error_type=ErrorType.RATE_LIMIT,
            status_code=429,
            message=str(error),
            retryable=True,
            retry_after=retry_after,
            original=error,
        )

    if status_code == 529:
        return ClassifiedError(
            error_type=ErrorType.SERVER_OVERLOAD,
            status_code=529,
            message=str(error),
            retryable=True,
            retry_after=1.0,
            original=error,
        )

    if status_code in (401, 403):
        return ClassifiedError(
            error_type=ErrorType.AUTH_ERROR,
            status_code=status_code,
            message=str(error),
            retryable=False,
            original=error,
        )

    if status_code == 400:
        # Check for max_tokens overflow
        overflow = parse_max_tokens_overflow(str(error))
        if overflow:
            return ClassifiedError(
                error_type=ErrorType.MAX_TOKENS_OVERFLOW,
                status_code=400,
                message=str(error),
                retryable=True,
                max_tokens_info=overflow,
                original=error,
            )

        if "model" in msg and ("not found" in msg or "invalid" in msg or "does not exist" in msg):
            return ClassifiedError(
                error_type=ErrorType.INVALID_MODEL,
                status_code=400,
                message=str(error),
                retryable=False,
                original=error,
            )

        if "too long" in msg or "too many tokens" in msg or "exceeds" in msg:
            return ClassifiedError(
                error_type=ErrorType.PROMPT_TOO_LONG,
                status_code=400,
                message=str(error),
                retryable=False,
                original=error,
            )

        return ClassifiedError(
            error_type=ErrorType.INVALID_REQUEST,
            status_code=400,
            message=str(error),
            retryable=False,
            original=error,
        )

    # ── By message content ────────────────────────────────────────────────

    if any(kw in msg for kw in _TIMEOUT_KEYWORDS):
        return ClassifiedError(
            error_type=ErrorType.API_TIMEOUT,
            status_code=status_code,
            message=str(error),
            retryable=True,
            retry_after=2.0,
            original=error,
        )

    if any(kw in msg for kw in _CONNECTION_KEYWORDS):
        return ClassifiedError(
            error_type=ErrorType.CONNECTION_ERROR,
            status_code=status_code,
            message=str(error),
            retryable=True,
            retry_after=1.0,
            original=error,
        )

    if "cancel" in msg or "abort" in msg:
        return ClassifiedError(
            error_type=ErrorType.ABORTED,
            status_code=status_code,
            message=str(error),
            retryable=False,
            original=error,
        )

    if "refus" in msg or "safety" in msg or "content policy" in msg:
        return ClassifiedError(
            error_type=ErrorType.REFUSAL,
            status_code=status_code,
            message=str(error),
            retryable=False,
            original=error,
        )

    # ── Server errors ─────────────────────────────────────────────────────

    if status_code and 500 <= status_code < 600:
        return ClassifiedError(
            error_type=ErrorType.SERVER_OVERLOAD,
            status_code=status_code,
            message=str(error),
            retryable=True,
            retry_after=2.0,
            original=error,
        )

    # ── Fallback ──────────────────────────────────────────────────────────

    return ClassifiedError(
        error_type=ErrorType.UNKNOWN,
        status_code=status_code,
        message=str(error),
        retryable=False,
        original=error,
    )


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
