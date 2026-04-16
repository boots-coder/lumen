"""Core type definitions for Lumen."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Literal


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool call requested by the model."""
    id: str  # Unique ID for this tool call
    name: str  # Tool name
    arguments: dict[str, Any]  # Tool arguments (already parsed)


@dataclass
class Message:
    role: Role
    content: str
    # Token count for this message (estimated or from API)
    token_count: int = 0
    # Metadata — preserved across compaction for audit
    created_at: datetime = field(default_factory=datetime.utcnow)
    # Marks a compaction boundary summary message
    is_compact_summary: bool = False

    # ── Tool calling support ──────────────────────────────────────────────
    # For assistant messages that contain tool calls
    tool_calls: list[ToolCall] | None = None
    # For tool result messages (role == TOOL)
    tool_call_id: str | None = None
    tool_name: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message format."""
        if self.role == Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content,
            }

        if self.role == Role.ASSISTANT and self.tool_calls:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": self.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                tc.arguments if isinstance(tc.arguments, str)
                                else __import__("json").dumps(tc.arguments)
                            ),
                        },
                    }
                    for tc in self.tool_calls
                ],
            }
            return msg

        return {"role": self.role.value, "content": self.content}

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to Anthropic Messages API format."""
        if self.role == Role.TOOL:
            # Anthropic: tool results are user messages with tool_result content blocks
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id or "",
                        "content": self.content,
                    }
                ],
            }

        if self.role == Role.ASSISTANT and self.tool_calls:
            # Anthropic: assistant messages with tool_use content blocks
            content_blocks: list[dict[str, Any]] = []
            if self.content:
                content_blocks.append({"type": "text", "text": self.content})
            for tc in self.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            return {"role": "assistant", "content": content_blocks}

        return {"role": self.role.value, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        """Default: OpenAI format (most common)."""
        return self.to_openai_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        role_str = data.get("role", "user")
        # Map "tool" role
        try:
            role = Role(role_str)
        except ValueError:
            role = Role.USER

        msg = cls(
            role=role,
            content=data.get("content", "") or "",
            token_count=data.get("token_count", 0),
            is_compact_summary=data.get("is_compact_summary", False),
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
        )

        # Restore tool_calls if present
        tc_data = data.get("tool_calls")
        if tc_data:
            msg.tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                )
                for tc in tc_data
            ]

        return msg


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Running token accounting for the session."""
    total: int = 0           # Total tokens in current context window
    prompt: int = 0          # Tokens used by all messages + system
    completion: int = 0      # Tokens used by last assistant response
    context_window: int = 0  # Model's max context window
    compaction_count: int = 0  # How many times we've compacted

    @property
    def percent_used(self) -> float:
        if self.context_window == 0:
            return 0.0
        return round(self.total / self.context_window * 100, 1)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.context_window - self.total)


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    api_key: str
    model: str
    base_url: str
    max_output_tokens: int
    context_window: int
    provider_type: Literal["openai", "anthropic"]
    # Optional separate model for compaction (can use a stronger model)
    compact_model: str | None = None
    compact_api_key: str | None = None
    compact_base_url: str | None = None


# ---------------------------------------------------------------------------
# Compaction result
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    summary: str                    # The structured summary text
    messages_before: int            # Message count before compaction
    messages_after: int             # Message count after compaction
    tokens_before: int
    tokens_after: int
    kept_recent_count: int = 0      # How many recent messages were kept verbatim


# ---------------------------------------------------------------------------
# Memory file info
# ---------------------------------------------------------------------------

class MemoryLayer(str, Enum):
    SYSTEM = "system"    # /etc/engram/ENGRAM.md
    USER = "user"        # ~/.engram/ENGRAM.md
    PROJECT = "project"  # ./ENGRAM.md, ./.engram/rules/*.md
    LOCAL = "local"      # ./ENGRAM.local.md


@dataclass
class MemoryFile:
    path: str
    layer: MemoryLayer
    content: str


# ---------------------------------------------------------------------------
# Provider response
# ---------------------------------------------------------------------------

@dataclass
class ProviderResponse:
    """Raw response from LLM provider (single turn)."""
    content: str | None  # Text content (may be None if only tool calls)
    tool_calls: list[ToolCall] | None  # Tool calls requested by model
    finish_reason: str  # "stop", "tool_calls", "length", etc.
    prompt_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# Chat response
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """Final response from Agent.chat() after tool execution."""
    content: str
    token_usage: TokenUsage
    was_compacted: bool = False
    compaction_result: CompactionResult | None = None
