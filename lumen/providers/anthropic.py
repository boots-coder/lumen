"""
Anthropic native provider.

Uses the Anthropic Messages API directly (no SDK dependency).
Handles Anthropic's unique message format where system is a top-level
parameter, not a message in the array.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from .base import BaseProvider
from .._types import ProviderResponse, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        return payload

    # ── BaseProvider implementation ───────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        payload = self._build_payload(messages, system, max_tokens, temperature, stream=False)

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/messages",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Parse content blocks
        content_text = ""
        tool_calls = []

        for block in data.get("content", []):
            if block.get("type") == "text":
                content_text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["input"],
                    )
                )

        # Finish reason
        finish_reason = data.get("stop_reason", "end_turn")
        if finish_reason == "end_turn" and tool_calls:
            finish_reason = "tool_use"

        # Token usage
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return ProviderResponse(
            content=content_text or None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, system, max_tokens, temperature, stream=True)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/messages",
                headers=self._headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text
                    elif event_type == "message_stop":
                        break
