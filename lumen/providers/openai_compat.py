"""
OpenAI-compatible provider.

Works with any API that speaks the OpenAI chat completions protocol:
  - OpenAI (api.openai.com)
  - Azure OpenAI
  - Together AI, Fireworks, Groq, Perplexity
  - Local via LM Studio, Ollama (openai-compat mode), vLLM
  - DeepSeek, Mistral, GetGoAPI, etc.

Covers roughly 90% of LLM APIs in the wild.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from .base import BaseProvider
from .._types import ProviderResponse, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_reasoning_model(self) -> bool:
        """o1 / o3 family: different API surface than chat models."""
        m = self.model.lower()
        return m.startswith(("o1", "o3")) or m in ("o1", "o3")

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        system: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        reasoning = self._is_reasoning_model()

        system_role = "developer" if self.model.lower().startswith("o1") else "system"

        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": system_role, "content": system})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "stream": stream,
        }

        if reasoning:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature

        if stream:
            payload["stream_options"] = {"include_usage": True}

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
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Parse response
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        # Extract content and tool calls
        content = message.get("content")
        tool_calls = None

        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = []
            for tc in message["tool_calls"]:
                # Parse arguments — may be string or dict
                args_raw = tc["function"]["arguments"]
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                else:
                    args = args_raw

                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args,
                    )
                )

        # Token usage
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
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
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    text = choices[0].get("delta", {}).get("content")
                    if text:
                        yield text
