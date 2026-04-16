"""
Thinking Support — extended reasoning / chain-of-thought budget control.

Supports:
  - Anthropic Claude thinking blocks (thinking: {"type": "enabled", "budget_tokens": N})
  - OpenAI o-series reasoning_effort parameter
  - Generic: prompt-based CoT injection

Lets users control how much "thinking" the model does per turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..providers.model_profiles import ModelProfile

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Types
# ═════════════════════════════════════════════════════════════════════════════

class ThinkingMode(str, Enum):
    """Controls when extended thinking is applied."""
    AUTO = "auto"              # Let the framework decide based on context
    ALWAYS = "always"          # Always enable thinking
    NEVER = "never"            # Never enable thinking
    BUDGET_ONLY = "budget_only"  # Enable only if budget allows


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking / reasoning."""
    enabled: bool = False
    budget_tokens: int = 10_000           # Max tokens for thinking
    mode: ThinkingMode = ThinkingMode.AUTO
    show_thinking: bool = False           # Whether to expose thinking to user


@dataclass
class ThinkingResult:
    """Parsed result separating thinking from final content."""
    thinking_text: str | None    # The reasoning/thinking content (if available)
    content: str                 # The final answer content
    thinking_tokens: int = 0     # Token count used for thinking


# ═════════════════════════════════════════════════════════════════════════════
# ThinkingManager
# ═════════════════════════════════════════════════════════════════════════════

class ThinkingManager:
    """
    Manages extended thinking / chain-of-thought across different model families.

    Handles three provider strategies:
      - Anthropic: native thinking blocks via the ``thinking`` request parameter
      - OpenAI o-series: ``reasoning_effort`` parameter
      - Generic: prompt-based CoT injection ("Think step by step")
    """

    # ── Request preparation ──────────────────────────────────────────────────

    @staticmethod
    def prepare_request(
        config: ThinkingConfig,
        profile: ModelProfile,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Inject thinking parameters into the API request payload.

        Modifies the payload in-place and returns it for convenience.

        Strategy depends on the model's thinking_format capability:
          - anthropic_blocks: add ``thinking`` parameter
          - openai_reasoning: set ``reasoning_effort``
          - none: prepend CoT instruction to system prompt
        """
        if not config.enabled and config.mode == ThinkingMode.NEVER:
            return payload

        effective_enabled = _should_enable(config)
        if not effective_enabled:
            return payload

        thinking_format = getattr(profile.capabilities, "thinking_format", "none")

        if thinking_format == "anthropic_blocks":
            payload = _prepare_anthropic(config, payload)
        elif thinking_format == "openai_reasoning":
            payload = _prepare_openai_reasoning(config, payload)
        else:
            payload = _prepare_generic_cot(payload)

        logger.debug(
            "Thinking prepared: format=%s budget=%d",
            thinking_format,
            config.budget_tokens,
        )
        return payload

    # ── Response parsing ─────────────────────────────────────────────────────

    @staticmethod
    def parse_thinking_response(
        response: Any,
        profile: ModelProfile,
    ) -> ThinkingResult:
        """
        Extract thinking content from the provider response.

        For models that support native thinking (Anthropic thinking blocks,
        o-series reasoning tokens), this separates the reasoning from the
        final answer. For generic models, thinking is not separable.
        """
        thinking_format = getattr(profile.capabilities, "thinking_format", "none")
        content = response.content or ""
        thinking_text = None
        thinking_tokens = 0

        if thinking_format == "anthropic_blocks":
            thinking_text, content, thinking_tokens = _parse_anthropic_thinking(response)
        elif thinking_format == "openai_reasoning":
            thinking_tokens = _parse_openai_reasoning_tokens(response)
            # o-series doesn't expose reasoning text, only token counts

        return ThinkingResult(
            thinking_text=thinking_text,
            content=content,
            thinking_tokens=thinking_tokens,
        )

    # ── Budget adjustment ────────────────────────────────────────────────────

    @staticmethod
    def adjust_budget(
        token_usage: Any,
        config: ThinkingConfig,
    ) -> ThinkingConfig:
        """
        Auto-adjust thinking budget based on context usage.

        When context is getting full, reduce the thinking budget to leave
        room for the actual response. Returns a new ThinkingConfig with
        the adjusted budget.
        """
        context_window = getattr(token_usage, "context_window", 0)
        total_used = getattr(token_usage, "total", 0)

        if context_window <= 0:
            return config

        percent_used = total_used / context_window

        if percent_used > 0.85:
            # Context nearly full — drastically reduce thinking
            new_budget = min(config.budget_tokens, 2_000)
            logger.debug(
                "Context %.0f%% full — reducing thinking budget to %d",
                percent_used * 100,
                new_budget,
            )
        elif percent_used > 0.70:
            # Getting tight — halve the budget
            new_budget = min(config.budget_tokens, config.budget_tokens // 2)
            logger.debug(
                "Context %.0f%% full — halving thinking budget to %d",
                percent_used * 100,
                new_budget,
            )
        else:
            new_budget = config.budget_tokens

        return ThinkingConfig(
            enabled=config.enabled,
            budget_tokens=new_budget,
            mode=config.mode,
            show_thinking=config.show_thinking,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _should_enable(config: ThinkingConfig) -> bool:
    """Determine whether thinking should be active for this turn."""
    if config.mode == ThinkingMode.NEVER:
        return False
    if config.mode == ThinkingMode.ALWAYS:
        return True
    if config.mode == ThinkingMode.BUDGET_ONLY:
        return config.budget_tokens > 0
    # AUTO — enabled if config says so
    return config.enabled


def _prepare_anthropic(config: ThinkingConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Add Anthropic thinking block parameter.

    Anthropic's extended thinking uses:
      thinking: {"type": "enabled", "budget_tokens": N}
    """
    payload["thinking"] = {
        "type": "enabled",
        "budget_tokens": config.budget_tokens,
    }
    # Anthropic requires temperature=1 when thinking is enabled
    payload["temperature"] = 1
    # Ensure max_tokens is large enough to hold thinking + response
    current_max = payload.get("max_tokens", 4096)
    if current_max < config.budget_tokens + 4096:
        payload["max_tokens"] = config.budget_tokens + 4096
    return payload


def _prepare_openai_reasoning(config: ThinkingConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Set OpenAI o-series reasoning_effort parameter.

    Maps budget tokens to reasoning_effort levels:
      - < 5000:  "low"
      - < 15000: "medium"
      - >= 15000: "high"
    """
    if config.budget_tokens < 5_000:
        effort = "low"
    elif config.budget_tokens < 15_000:
        effort = "medium"
    else:
        effort = "high"

    payload["reasoning_effort"] = effort
    # o-series uses max_completion_tokens instead of max_tokens
    if "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
    return payload


def _prepare_generic_cot(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Inject chain-of-thought instruction for models without native thinking support.

    Prepends "Think step by step before answering." to the system message.
    """
    cot_prefix = "Think step by step before answering.\n\n"
    messages = payload.get("messages", [])
    if messages and messages[0].get("role") in ("system", "developer"):
        messages[0]["content"] = cot_prefix + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": cot_prefix.strip()})
    payload["messages"] = messages
    return payload


def _parse_anthropic_thinking(response: Any) -> tuple[str | None, str, int]:
    """
    Parse Anthropic thinking blocks from the response.

    Anthropic returns thinking as separate content blocks with type="thinking".
    Falls back to checking the response content if no structured blocks found.

    Returns (thinking_text, final_content, thinking_tokens).
    """
    # Check for structured thinking blocks in raw response data
    raw_content = getattr(response, "_raw_content", None)
    if raw_content and isinstance(raw_content, list):
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        thinking_tokens = 0

        for block in raw_content:
            if isinstance(block, dict):
                if block.get("type") == "thinking":
                    thinking_parts.append(block.get("thinking", ""))
                elif block.get("type") == "text":
                    content_parts.append(block.get("text", ""))

        if thinking_parts:
            return (
                "\n".join(thinking_parts),
                "\n".join(content_parts) if content_parts else (response.content or ""),
                getattr(response, "thinking_tokens", 0),
            )

    # Fallback: no structured thinking blocks found
    return None, response.content or "", 0


def _parse_openai_reasoning_tokens(response: Any) -> int:
    """
    Extract reasoning token count from o-series response.

    The o-series models report reasoning tokens in usage.completion_tokens_details.
    """
    # Check for reasoning tokens in usage details
    usage = getattr(response, "_raw_usage", None)
    if usage and isinstance(usage, dict):
        details = usage.get("completion_tokens_details", {})
        if isinstance(details, dict):
            return details.get("reasoning_tokens", 0)
    return 0
