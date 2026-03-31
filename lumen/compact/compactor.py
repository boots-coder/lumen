"""
Conversation compactor.

Orchestrates the full compaction flow:
  1. Build a compaction request (recent N messages → full history)
  2. Call the model to generate a structured summary
  3. Parse the summary from the model response
  4. Replace the session's message history with:
       [compact_summary_message] + [kept_recent_messages]
  5. Recalculate token counts

Two compaction modes:
  - BASE compaction  : entire history → one summary message
  - PARTIAL compaction: keep the last K messages verbatim, summarise everything before
  - Circuit breaker  : stop after MAX_CONSECUTIVE_FAILURES consecutive failures

Partial compaction strategy:
  We keep the most recent `keep_recent` messages intact so the model retains
  immediate context (the last task, last code snippet, etc.) without having
  to re-derive it from the summary. See also:
  
"""

from __future__ import annotations

import logging
from typing import Any

from .._types import CompactionResult, Message, Role
from ..context.session import Session
from ..tokens.counter import count_messages_tokens, count_tokens
from .auto_compact import MAX_CONSECUTIVE_FAILURES
from .prompt import (
    BASE_COMPACT_PROMPT,
    PARTIAL_COMPACT_PROMPT,
    build_compact_user_message,
    extract_summary,
)

logger = logging.getLogger(__name__)

# How many recent messages to keep verbatim in partial compaction.
# Uses a simple fixed keep_recent count.
DEFAULT_KEEP_RECENT = 6


class Compactor:
    """
    Stateless compaction engine.

    Instantiate once and call compact() whenever the auto-compact threshold
    is reached. The Compactor does not mutate the Session directly — it returns
    a CompactionResult and lets the caller (Agent) apply it.
    """

    def __init__(
        self,
        provider,  # BaseProvider — injected by Agent to avoid circular import
        model: str,
        max_output_tokens: int = 8_000,
        keep_recent: int = DEFAULT_KEEP_RECENT,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._keep_recent = keep_recent

    # ── public API ────────────────────────────────────────────────────────────

    async def compact(
        self,
        session: Session,
        system_prompt: str,
        partial: bool = True,
    ) -> CompactionResult:
        """
        Compact the session's message history.

        Args:
            session:       The live session to compact.
            system_prompt: Current system prompt (used for token recalculation).
            partial:       If True, keep the most recent messages verbatim and
                           only summarise earlier history. If False, summarise
                           the entire conversation.

        Returns:
            CompactionResult with before/after statistics.

        Raises:
            RuntimeError if compaction fails after MAX_CONSECUTIVE_FAILURES.
        """
        messages = session.messages
        tokens_before = session.token_usage.total
        msg_count_before = len(messages)

        if not messages:
            logger.warning("compact() called on empty session — nothing to do.")
            return CompactionResult(
                summary="",
                messages_before=0,
                messages_after=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        # ── Choose compaction mode ────────────────────────────────────────────
        if partial and len(messages) > self._keep_recent:
            to_summarise = messages[: -self._keep_recent]
            to_keep = messages[-self._keep_recent :]
            prompt_template = PARTIAL_COMPACT_PROMPT
            logger.info(
                "Partial compaction: summarising %d messages, keeping %d recent.",
                len(to_summarise), len(to_keep),
            )
        else:
            to_summarise = messages
            to_keep = []
            prompt_template = BASE_COMPACT_PROMPT
            logger.info("Full compaction: summarising all %d messages.", len(to_summarise))

        # ── Call the model ────────────────────────────────────────────────────
        summary_raw = await self._call_compact_model(to_summarise, prompt_template)

        # ── Parse summary ─────────────────────────────────────────────────────
        summary = extract_summary(summary_raw)
        compact_message_text = build_compact_user_message(summary)

        # ── Rebuild message list ──────────────────────────────────────────────
        compact_msg = Message(
            role=Role.USER,
            content=compact_message_text,
            token_count=count_tokens(compact_message_text, self._model),
            is_compact_summary=True,
        )

        new_messages: list[Message] = [compact_msg] + list(to_keep)

        # ── Update session ────────────────────────────────────────────────────
        new_total = session.recalculate_tokens_for(new_messages, system_prompt)
        session.replace_messages(new_messages, new_total)

        logger.info(
            "Compaction complete: %d → %d messages, %d → %d tokens.",
            msg_count_before, len(new_messages), tokens_before, new_total,
        )

        return CompactionResult(
            summary=summary,
            messages_before=msg_count_before,
            messages_after=len(new_messages),
            tokens_before=tokens_before,
            tokens_after=new_total,
            kept_recent_count=len(to_keep),
        )

    # ── internal ──────────────────────────────────────────────────────────────

    async def _call_compact_model(
        self,
        messages_to_summarise: list[Message],
        prompt_template: str,
    ) -> str:
        """
        Send the compaction request to the model.

        The compaction prompt is passed as the system message.
        The conversation history to summarise is the user/assistant messages.
        We ask the model to produce the summary in a single response
        (no tool calls, no streaming — we want the full output).
        """
        api_messages = [m.to_dict() for m in messages_to_summarise]

        # Append a final user message asking for the summary
        # (required for models that need a user turn to respond)
        api_messages.append({
            "role": "user",
            "content": (
                "Please create a detailed summary of this conversation following "
                "the instructions in the system prompt."
            ),
        })

        text, _, _ = await self._provider.chat(
            messages=api_messages,
            system=prompt_template,
            max_tokens=self._max_output_tokens,
            temperature=0.0,  # deterministic for consistent summaries
        )
        return text


# ── Session extension helper ──────────────────────────────────────────────────
# We add recalculate_tokens_for() as a standalone function rather than patching
# Session, to keep Session free of compact-module imports.

def recalculate_tokens_for(
    session: Session,
    messages: list[Message],
    system_prompt: str,
) -> int:
    """Recalculate the total token count for a given message list + system prompt."""
    all_msgs = [{"role": "system", "content": system_prompt}]
    all_msgs += [m.to_dict() for m in messages]
    total = count_messages_tokens(all_msgs, session.model)
    return total


# Monkey-patch onto Session so Compactor can call session.recalculate_tokens_for()
def _patch_session() -> None:
    from ..context.session import Session

    def _method(self: Session, messages: list[Message], system_prompt: str) -> int:
        return recalculate_tokens_for(self, messages, system_prompt)

    Session.recalculate_tokens_for = _method  # type: ignore[attr-defined]


_patch_session()
