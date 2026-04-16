"""
Token counting — model-agnostic with graceful fallbacks.

Strategy (in order):
  1. tiktoken  — accurate for OpenAI models and a good approximation for others
  2. Character approximation — 4 chars ≈ 1 token (universal fallback)

Strategy: tiktoken for OpenAI models, character approximation as fallback.



"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ── Known context windows ────────────────────────────────────────────────────
# Known model registry.
# Users can always override via Agent(context_window=N).
KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4.1-2025-04-14": 1_047_576,
    "gpt-4.1-mini-2025-04-14": 1_047_576,
    "gpt-4.1-nano-2025-04-14": 1_047_576,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    # Anthropic
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    # Google
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    # Meta / Ollama common
    "llama3": 8_192,
    "llama3.1": 128_000,
    "llama3.2": 128_000,
    "mistral": 32_768,
    "mixtral": 32_768,
    "qwen2.5": 128_000,
    "deepseek-r1": 64_000,
    "phi3": 128_000,
}

# Known max output tokens per model
KNOWN_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "gpt-4.1-2025-04-14": 32_768,
    "gpt-4.1-mini-2025-04-14": 32_768,
    "gpt-4.1-nano-2025-04-14": 32_768,
    "gpt-4.1": 32_768,
    "gpt-4.1-mini": 32_768,
    "gpt-4.1-nano": 32_768,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-4-turbo": 4_096,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 4_096,
    "o1": 32_768,
    "o1-mini": 65_536,
    "claude-opus-4-6": 64_000,
    "claude-sonnet-4-6": 32_000,
    "claude-haiku-4-5": 32_000,
    "claude-3-5-sonnet-20241022": 8_096,
    "claude-3-5-haiku-20241022": 8_096,
    "claude-3-opus-20240229": 4_096,
    "gemini-1.5-pro": 8_192,
    "gemini-1.5-flash": 8_192,
}

_DEFAULT_CONTEXT_WINDOW = 8_192
_DEFAULT_MAX_OUTPUT_TOKENS = 4_096

# ── tiktoken encoding cache ──────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _get_tiktoken_encoding(model: str):
    """Return a tiktoken Encoding for the given model, or None if unavailable."""
    try:
        import tiktoken  # type: ignore
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base — works well for GPT-4 family and is a
            # reasonable approximation for most modern models.
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


# ── Public helpers ───────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a string. Falls back to char approximation if tiktoken unavailable."""
    if not text:
        return 0
    enc = _get_tiktoken_encoding(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: 4 characters ≈ 1 token
    return max(1, len(text) // 4)


def count_messages_tokens(messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
    """
    Estimate total tokens for a list of message dicts.

    Uses OpenAI's counting formula (3 tokens overhead per message + content).
    This is a good approximation for all chat models.
    """
    total = 0
    for msg in messages:
        total += 3  # per-message overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            # Multi-part content blocks
            for block in content:
                if isinstance(block, dict):
                    total += count_tokens(block.get("text", ""), model)
        role = msg.get("role", "")
        total += count_tokens(role, model)
    total += 3  # reply priming
    return total


def get_context_window(model: str) -> int:
    """Return context window for a model. Tries prefix matching for versioned names."""
    if model in KNOWN_CONTEXT_WINDOWS:
        return KNOWN_CONTEXT_WINDOWS[model]
    # Prefix match for versioned model names
    for known, window in KNOWN_CONTEXT_WINDOWS.items():
        if model.startswith(known) or known.startswith(model):
            return window
    logger.warning(
        "Unknown model '%s' — using default context window of %d tokens. "
        "Pass context_window= to Agent() to override.",
        model,
        _DEFAULT_CONTEXT_WINDOW,
    )
    return _DEFAULT_CONTEXT_WINDOW


def get_max_output_tokens(model: str) -> int:
    """Return max output tokens for a model."""
    if model in KNOWN_MAX_OUTPUT_TOKENS:
        return KNOWN_MAX_OUTPUT_TOKENS[model]
    for known, max_out in KNOWN_MAX_OUTPUT_TOKENS.items():
        if model.startswith(known) or known.startswith(model):
            return max_out
    return _DEFAULT_MAX_OUTPUT_TOKENS
