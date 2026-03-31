"""
Provider factory — auto-selects the right provider from model name / base_url.

Detection order:
  1. Explicit base_url containing "anthropic.com"  → AnthropicProvider
  2. Model name starts with "claude"               → AnthropicProvider
  3. Everything else                               → OpenAICompatProvider
     (OpenAI, Azure, Together, Groq, Ollama, LM Studio, vLLM, …)
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import BaseProvider
from .openai_compat import OpenAICompatProvider

_ANTHROPIC_BASE = "https://api.anthropic.com"
_OPENAI_BASE = "https://api.openai.com/v1"


def create_provider(
    api_key: str,
    model: str,
    base_url: str | None,
    timeout: float = 120.0,
) -> tuple[BaseProvider, str]:
    """
    Return (provider_instance, resolved_base_url).

    base_url=None means "use the default for this model".
    """
    is_anthropic = (
        (base_url is not None and "anthropic.com" in base_url)
        or model.startswith("claude")
    )

    if is_anthropic:
        resolved = base_url or _ANTHROPIC_BASE
        return AnthropicProvider(api_key, model, resolved, timeout), resolved
    else:
        resolved = base_url or _OPENAI_BASE
        return OpenAICompatProvider(api_key, model, resolved, timeout), resolved
