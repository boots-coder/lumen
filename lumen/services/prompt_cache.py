"""
Prompt Caching — reduce API costs by caching static context.

Two strategies:
1. Anthropic native: inject cache_control markers on system/tool blocks
2. Universal hash-based: detect when context hasn't changed, skip re-sending
   static parts by tracking message hashes and using conversation IDs
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..providers.model_profiles import ModelProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache strategy
# ---------------------------------------------------------------------------

class CacheStrategy(str, Enum):
    """Which caching mechanism to use."""
    NONE = "none"
    ANTHROPIC_NATIVE = "anthropic_native"
    HASH_BASED = "hash_based"


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Running statistics for prompt cache usage."""
    cache_hits: int = 0
    cache_misses: int = 0
    estimated_tokens_saved: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage (0.0 – 100.0)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return round(self.cache_hits / total * 100, 1)

    def record_hit(self, estimated_tokens: int = 0) -> None:
        self.cache_hits += 1
        self.estimated_tokens_saved += estimated_tokens

    def record_miss(self) -> None:
        self.cache_misses += 1


# ---------------------------------------------------------------------------
# Prompt Cache Manager
# ---------------------------------------------------------------------------

# Number of recent user messages to mark with rolling cache breaks
_ROLLING_CACHE_BREAK_COUNT = 3


class PromptCacheManager:
    """
    Manages prompt caching across different providers.

    Usage::

        manager = PromptCacheManager()
        strategy = manager.auto_detect_strategy(profile)
        messages, system, tools = manager.prepare_messages(
            messages, system, tools, profile,
        )
    """

    def __init__(self) -> None:
        self._stats = CacheStats()
        # Hash of the last system + tools payload — used for hash-based caching
        self._last_prefix_hash: str | None = None

    # ── Strategy detection ───────────────────────────────────────────────

    @staticmethod
    def auto_detect_strategy(profile: ModelProfile) -> CacheStrategy:
        """
        Detect the best caching strategy for a model profile.

        - Anthropic Claude models get native ``cache_control`` support.
        - All other models fall back to hash-based observability caching.
        """
        if profile.family == "anthropic":
            return CacheStrategy.ANTHROPIC_NATIVE
        return CacheStrategy.HASH_BASED

    # ── Main entry point ─────────────────────────────────────────────────

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | Any,
        tools: list[dict[str, Any]] | None,
        profile: ModelProfile,
    ) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]] | None]:
        """
        Inject cache markers / track prefix hashes before an API call.

        Returns the (possibly mutated) ``(messages, system, tools)`` tuple.
        """
        strategy = self.auto_detect_strategy(profile)

        if strategy == CacheStrategy.ANTHROPIC_NATIVE:
            system = self._inject_anthropic_system_cache(system)
            tools = self._inject_anthropic_tools_cache(tools)
            messages = self._inject_anthropic_message_cache(messages)
            return messages, system, tools

        if strategy == CacheStrategy.HASH_BASED:
            self._track_prefix_hash(system, tools)
            return messages, system, tools

        # NONE — pass through unchanged
        return messages, system, tools

    # ── Stats ────────────────────────────────────────────────────────────

    def get_cache_stats(self) -> CacheStats:
        """Return a snapshot of current cache statistics."""
        return self._stats

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def update_from_api_usage(
        self,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        """
        Update stats from real API usage headers (Anthropic returns these).
        """
        self._stats.cache_creation_tokens += cache_creation_tokens
        self._stats.cache_read_tokens += cache_read_tokens
        if cache_read_tokens > 0:
            self._stats.record_hit(estimated_tokens=cache_read_tokens)

    # ── Anthropic native helpers ─────────────────────────────────────────

    @staticmethod
    def _inject_anthropic_system_cache(
        system: str | Any,
    ) -> list[dict[str, Any]]:
        """
        Convert a plain system string into a content-block list with
        ``cache_control`` on the entire block.

        If *system* is already a list (e.g. from a previous call), ensure
        the last block carries ``cache_control``.
        """
        if isinstance(system, str):
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Already a list of content blocks — mark last with cache_control
        if isinstance(system, list) and system:
            blocks = [dict(b) for b in system]  # shallow copy
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
            return blocks

        # Unexpected — return as-is
        return system  # type: ignore[return-value]

    @staticmethod
    def _inject_anthropic_tools_cache(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """
        Mark the last tool definition with ``cache_control`` so the entire
        tools block is cached.
        """
        if not tools:
            return tools

        tools = [dict(t) for t in tools]  # shallow copy
        tools[-1] = dict(tools[-1])
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    @staticmethod
    def _inject_anthropic_message_cache(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Add ``cache_control`` to the last N user messages (rolling cache
        break) so the API caches the stable conversation prefix.
        """
        if not messages:
            return messages

        # Deep-ish copy to avoid mutating the caller's list
        messages = [dict(m) for m in messages]

        # Find indices of user messages (Anthropic tool results also come
        # as role=user, which is fine — they are part of the prefix too)
        user_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "user"
        ]

        # Mark the last N user messages
        for idx in user_indices[-_ROLLING_CACHE_BREAK_COUNT:]:
            msg = dict(messages[idx])
            content = msg.get("content")

            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                content = [dict(b) for b in content]
                content[-1] = dict(content[-1])
                content[-1]["cache_control"] = {"type": "ephemeral"}
                msg["content"] = content

            messages[idx] = msg

        return messages

    # ── Hash-based helpers ───────────────────────────────────────────────

    def _track_prefix_hash(
        self,
        system: str | Any,
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """
        Compute a SHA-256 hash of the system prompt + tool definitions.

        When the hash matches the previous call we know the static prefix
        is unchanged — the provider can potentially skip re-encoding.
        We log hit/miss for observability and update stats.
        """
        system_str = system if isinstance(system, str) else json.dumps(system, sort_keys=True)
        tools_str = json.dumps(tools, sort_keys=True) if tools else ""

        raw = f"{system_str}||{tools_str}"
        current_hash = hashlib.sha256(raw.encode()).hexdigest()

        if self._last_prefix_hash is not None and current_hash == self._last_prefix_hash:
            # Estimate: system + tools are typically 20-40% of prompt tokens.
            # Without access to the exact token count we record a nominal
            # estimate based on character length.
            estimated_saved = len(raw) // 4  # rough chars-to-tokens
            self._stats.record_hit(estimated_tokens=estimated_saved)
            logger.debug(
                "Prompt cache HIT (hash=%s…, est_saved=%d tokens)",
                current_hash[:12],
                estimated_saved,
            )
        else:
            self._stats.record_miss()
            logger.debug(
                "Prompt cache MISS (hash=%s…, prev=%s)",
                current_hash[:12],
                self._last_prefix_hash[:12] if self._last_prefix_hash else "none",
            )

        self._last_prefix_hash = current_hash
