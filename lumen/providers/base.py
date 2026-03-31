"""Abstract provider interface — all LLM providers implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

# Import ProviderResponse from _types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _types import ProviderResponse


class BaseProvider(ABC):
    """
    Thin adapter between Engram's message format and a specific LLM API.

    Each provider normalises:
      - Request construction  (messages → API payload)
      - Response parsing      (API response → ProviderResponse)
      - Streaming             (SSE / chunked → AsyncIterator[str])
      - Tool calling          (tools → API format, tool_calls parsing)

    Engram's context management sits entirely above this layer and never
    touches provider-specific details.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        """
        Send messages and return provider response.

        Args:
            messages: List of message dicts (role, content).
            system:   System prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Optional list of tool schemas (OpenAI or Anthropic format).

        Returns:
            ProviderResponse with content, tool_calls, and token counts.
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """
        Stream response text chunks.

        Yields raw text deltas as they arrive from the API.
        Note: Streaming does not support tool calls yet.
        """
        # mypy requires an explicit yield in abstract async generators
        yield  # type: ignore[misc]
