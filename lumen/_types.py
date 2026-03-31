"""Core type definitions for Engram."""

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

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            token_count=data.get("token_count", 0),
            is_compact_summary=data.get("is_compact_summary", False),
        )


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
# Tool calling
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A tool call requested by the model."""
    id: str  # Unique ID for this tool call
    name: str  # Tool name
    arguments: dict[str, Any]  # Tool arguments (already parsed)


# ---------------------------------------------------------------------------
# Chat response
# ---------------------------------------------------------------------------

@dataclass
class ProviderResponse:
    """Raw response from LLM provider (single turn)."""
    content: str | None  # Text content (may be None if only tool calls)
    tool_calls: list[ToolCall] | None  # Tool calls requested by model
    finish_reason: str  # "stop", "tool_calls", "length", etc.
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ChatResponse:
    """Final response from Agent.chat() after tool execution."""
    content: str
    token_usage: TokenUsage
    was_compacted: bool = False
    compaction_result: CompactionResult | None = None
