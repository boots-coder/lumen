"""
Auto-Dream — background memory consolidation.

Every N turns we fire-and-forget two pieces of work:

  1. **Extraction** (reuses SessionMemoryManager.extract_from_turn) —
     pulls fresh facts out of the most recent conversation slice.
  2. **Consolidation** (every `consolidation_every` dreams) — asks the
     model to merge accumulated entries into a short narrative summary,
     appended back as a high-relevance `PATTERN` entry so the session
     retains long-horizon structure even across compactions.

Both steps run inside an asyncio task so the user never waits for them;
exceptions are swallowed and logged. Design is pull-based: the agent
calls `on_turn_complete()` after each `chat()`; the service decides
whether to dream and returns immediately.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..context.session_memory import MemoryCategory, MemoryEntry

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


_CONSOLIDATION_PROMPT = """\
You are a memory consolidation system. Below are discrete facts extracted \
from an ongoing coding session with the user. Your job: merge them into ONE \
coherent narrative summary (3-8 sentences) capturing what matters for future \
turns. Prioritize:
  · project context (what's being built, architecture decisions)
  · user preferences and corrections
  · open threads / in-progress work
  · key file paths

Do NOT enumerate — write flowing prose. Omit trivial duplicates. Keep it <= 400 words.

Facts to consolidate:
"""


@dataclass
class AutoDreamConfig:
    enabled: bool = True
    interval_turns: int = 3          # extract every N turns
    consolidation_every: int = 4     # consolidate every K extractions
    max_entries_before_forced_consolidation: int = 60


@dataclass
class DreamStats:
    extractions: int = 0
    consolidations: int = 0
    last_error: str | None = None
    last_run_at: str | None = None
    currently_dreaming: bool = False


class AutoDreamService:
    """Background memory consolidation. Install with `agent.enable_auto_dream()`."""

    def __init__(self, agent: Agent, config: AutoDreamConfig | None = None) -> None:
        self._agent = agent
        self._config = config or AutoDreamConfig()
        self._stats = DreamStats()
        self._active_task: asyncio.Task | None = None
        self._turns_since_last_dream = 0

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def config(self) -> AutoDreamConfig:
        return self._config

    @property
    def stats(self) -> DreamStats:
        return self._stats

    def on_turn_complete(self) -> None:
        """Called by Agent.chat() after each turn. Non-blocking."""
        if not self._config.enabled:
            return
        if self._agent._session_memory is None:
            return
        self._turns_since_last_dream += 1
        if self._turns_since_last_dream < self._config.interval_turns:
            return
        # Don't stack tasks — if one's still running, skip.
        if self._active_task is not None and not self._active_task.done():
            logger.debug("Previous dream task still running; skipping this turn.")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._turns_since_last_dream = 0
        self._active_task = loop.create_task(self._dream(), name="auto-dream")

    async def force_dream(self) -> DreamStats:
        """Run a dream pass right now (awaits). Useful for /dream force."""
        await self._dream()
        return self._stats

    async def consolidate_now(self) -> str | None:
        """Run the consolidation step directly. Returns the narrative summary,
        or None if nothing to consolidate."""
        return await self._consolidate()

    # ── internals ────────────────────────────────────────────────────────────

    async def _dream(self) -> None:
        self._stats.currently_dreaming = True
        try:
            mem = self._agent._session_memory
            if mem is None:
                return
            # 1. extraction
            try:
                await mem.extract_from_turn(
                    self._agent._session.messages,
                    self._agent._provider,
                    self._agent._model,
                )
                self._stats.extractions += 1
            except Exception as e:
                logger.debug("Dream extraction failed: %s", e)
                self._stats.last_error = f"extract: {e}"

            # 2. consolidation (periodic, or forced if entries overflow)
            trigger_by_count = (
                self._stats.extractions > 0
                and self._stats.extractions % self._config.consolidation_every == 0
            )
            trigger_by_overflow = (
                len(mem) >= self._config.max_entries_before_forced_consolidation
            )
            if trigger_by_count or trigger_by_overflow:
                try:
                    await self._consolidate()
                except Exception as e:
                    logger.debug("Dream consolidation failed: %s", e)
                    self._stats.last_error = f"consolidate: {e}"

            # 3. persist
            try:
                mem.save()
            except Exception as e:
                logger.debug("Dream save failed: %s", e)

            self._stats.last_run_at = datetime.utcnow().isoformat()
        finally:
            self._stats.currently_dreaming = False

    async def _consolidate(self) -> str | None:
        mem = self._agent._session_memory
        if mem is None or len(mem) == 0:
            return None

        entries = list(mem._entries)
        if len(entries) < 4:
            return None  # not enough signal to consolidate

        bullets = "\n".join(
            f"- [{e.category.value}] {e.content}"
            for e in entries[-40:]  # cap prompt size
        )
        prompt = _CONSOLIDATION_PROMPT + bullets

        response = await self._agent._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You consolidate structured facts into flowing narrative. "
                "Be precise and dense."
            ),
            max_tokens=600,
            temperature=0.2,
        )
        summary = (response.content or "").strip()
        if not summary:
            return None

        # Store as a high-relevance pattern entry so it floats to the top
        mem._entries.append(MemoryEntry(
            content=f"[CONSOLIDATED] {summary}",
            category=MemoryCategory.PATTERN,
            relevance_score=0.95,
            source_turn=mem.turn_count,
        ))
        self._stats.consolidations += 1
        return summary
