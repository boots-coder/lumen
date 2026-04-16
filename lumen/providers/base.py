"""Abstract provider interface — all LLM providers implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from .._types import ProviderResponse


class BaseProvider(ABC):
    """
    Thin adapter between Lumen's message format and a specific LLM API.

    Each provider normalises:
      - Request construction  (messages → API payload)
      - Response parsing      (API response → ProviderResponse)
      - Streaming             (SSE / chunked → AsyncIterator[str])
      - Tool calling          (tools → API format, tool_calls parsing)
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
            messages: List of message dicts (already formatted for this provider).
            system:   System prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Optional list of tool schemas.

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
        """Stream response text chunks."""
        yield  # type: ignore[misc]
