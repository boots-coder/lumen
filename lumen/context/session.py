"""
Session — the heart of Lumen's context management.

Tracks the full message history, running token counts, and compaction state.

Key design decisions:
  - Token counts are tracked per-message and as a running total
  - Tool call messages use proper role formats (role=tool for OpenAI, user+tool_result for Anthropic)
  - Sessions are serialisable to JSON for persistence across processes
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .._types import Message, Role, TokenUsage, ToolCall
from ..tokens.counter import count_tokens, count_messages_tokens

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Mutable conversation state.

    messages        — the live message list sent to the API each turn
    token_usage     — running token accounting
    compaction_count — total number of compactions performed
    consecutive_compact_failures — circuit-breaker counter (reset on success)
    created_at      — ISO timestamp of session creation
    """

    model: str
    messages: list[Message] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    compaction_count: int = 0
    consecutive_compact_failures: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── message management ────────────────────────────────────────────────────

    def add_user(self, content: str) -> Message:
        tokens = count_tokens(content, self.model)
        msg = Message(role=Role.USER, content=content, token_count=tokens)
        self.messages.append(msg)
        self.token_usage.prompt += tokens
        self.token_usage.total += tokens
        return msg

    def add_assistant(self, content: str, completion_tokens: int = 0) -> Message:
        tokens = completion_tokens or count_tokens(content, self.model)
        msg = Message(role=Role.ASSISTANT, content=content, token_count=tokens)
        self.messages.append(msg)
        self.token_usage.completion = tokens
        self.token_usage.total += tokens
        return msg

    def add_assistant_with_tool_calls(
        self,
        content: str | None,
        tool_calls: list[ToolCall],
        completion_tokens: int = 0,
    ) -> Message:
        """Add an assistant message that includes tool calls."""
        text = content or ""
        tokens = completion_tokens or count_tokens(text, self.model) + 50 * len(tool_calls)
        msg = Message(
            role=Role.ASSISTANT,
            content=text,
            token_count=tokens,
            tool_calls=tool_calls,
        )
        self.messages.append(msg)
        self.token_usage.completion = tokens
        self.token_usage.total += tokens
        return msg

    def add_tool_result(
        self,
        tool_use_id: str,
        tool_name: str,
        content: str,
        is_error: bool = False,
    ) -> Message:
        """
        Add a tool result message.

        Uses Role.TOOL with tool_call_id — providers convert to the right
        wire format (OpenAI: role=tool, Anthropic: role=user + tool_result block).
        """
        if is_error:
            content = f"Error: {content}"

        tokens = count_tokens(content, self.model)
        msg = Message(
            role=Role.TOOL,
            content=content,
            token_count=tokens,
            tool_call_id=tool_use_id,
            tool_name=tool_name,
        )
        self.messages.append(msg)
        self.token_usage.total += tokens
        return msg

    def add_compact_summary(self, summary: str) -> Message:
        """Insert a compaction boundary message."""
        tokens = count_tokens(summary, self.model)
        msg = Message(
            role=Role.USER,
            content=summary,
            token_count=tokens,
            is_compact_summary=True,
        )
        self.messages.append(msg)
        self.token_usage.total += tokens
        return msg

    def replace_messages(self, new_messages: list[Message], new_total_tokens: int) -> None:
        """Replace the message list after compaction and recalculate totals."""
        self.messages = new_messages
        self.token_usage.total = new_total_tokens
        self.token_usage.prompt = new_total_tokens
        self.compaction_count += 1
        self.consecutive_compact_failures = 0

    def update_token_counts_from_api(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        context_window: int,
    ) -> None:
        """Update token counts using real numbers from the API response."""
        self.token_usage.prompt = prompt_tokens
        self.token_usage.completion = completion_tokens
        self.token_usage.total = prompt_tokens + completion_tokens
        self.token_usage.context_window = context_window

    # ── message format conversion ────────────────────────────────────────────

    def as_openai_messages(self) -> list[dict[str, Any]]:
        """Return messages in OpenAI-compatible format."""
        return [m.to_openai_dict() for m in self.messages]

    def as_anthropic_messages(self) -> list[dict[str, Any]]:
        """Return messages in Anthropic Messages API format."""
        messages = []
        for m in self.messages:
            d = m.to_anthropic_dict()
            # Anthropic requires alternating user/assistant roles.
            # Merge consecutive same-role messages.
            if messages and messages[-1]["role"] == d["role"]:
                prev = messages[-1]
                # Merge content
                if isinstance(prev["content"], list) and isinstance(d["content"], list):
                    prev["content"].extend(d["content"])
                elif isinstance(prev["content"], list):
                    prev["content"].append({"type": "text", "text": d["content"]})
                elif isinstance(d["content"], list):
                    prev["content"] = [{"type": "text", "text": prev["content"]}] + d["content"]
                else:
                    prev["content"] = prev["content"] + "\n\n" + d["content"]
            else:
                messages.append(d)
        return messages

    def as_api_messages(self, provider_type: str = "openai") -> list[dict[str, Any]]:
        """Return messages formatted for the given provider type."""
        if provider_type == "anthropic":
            return self.as_anthropic_messages()
        return self.as_openai_messages()

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "created_at": self.created_at,
            "compaction_count": self.compaction_count,
            "consecutive_compact_failures": self.consecutive_compact_failures,
            "token_usage": {
                "total": self.token_usage.total,
                "prompt": self.token_usage.prompt,
                "completion": self.token_usage.completion,
                "context_window": self.token_usage.context_window,
                "compaction_count": self.compaction_count,
            },
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "token_count": m.token_count,
                    "created_at": m.created_at.isoformat(),
                    "is_compact_summary": m.is_compact_summary,
                    "tool_call_id": m.tool_call_id,
                    "tool_name": m.tool_name,
                    "tool_calls": (
                        [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                         for tc in m.tool_calls]
                        if m.tool_calls else None
                    ),
                }
                for m in self.messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        session = cls(model=data["model"])
        session.created_at = data.get("created_at", session.created_at)
        session.compaction_count = data.get("compaction_count", 0)
        session.consecutive_compact_failures = data.get("consecutive_compact_failures", 0)

        usage = data.get("token_usage", {})
        session.token_usage = TokenUsage(
            total=usage.get("total", 0),
            prompt=usage.get("prompt", 0),
            completion=usage.get("completion", 0),
            context_window=usage.get("context_window", 0),
            compaction_count=data.get("compaction_count", 0),
        )
        for m in data.get("messages", []):
            msg = Message.from_dict(m)
            session.messages.append(msg)
        return session

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        logger.debug("Session saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        data = json.loads(Path(path).read_text())
        session = cls.from_dict(data)
        logger.debug("Session loaded from %s (%d messages)", path, len(session.messages))
        return session

    # ── convenience ───────────────────────────────────────────────────────────

    def recalculate_tokens(self, system: str) -> int:
        """Re-count tokens for the full context (system + messages)."""
        all_msgs = [{"role": "system", "content": system}] + self.as_openai_messages()
        total = count_messages_tokens(all_msgs, self.model)
        self.token_usage.total = total
        self.token_usage.prompt = total
        return total

    def __repr__(self) -> str:
        return (
            f"Session(model={self.model!r}, messages={len(self.messages)}, "
            f"tokens={self.token_usage.total}/{self.token_usage.context_window}, "
            f"compactions={self.compaction_count})"
        )
