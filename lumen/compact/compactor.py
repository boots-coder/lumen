"""
Conversation compactor.

Orchestrates the full compaction flow:
  1. Build a compaction request (recent N messages → full history)
  2. Call the model to generate a structured summary
  3. Parse the summary from the model response
  4. Replace the session's message history with:
       [compact_summary_message] + [kept_recent_messages]
  5. Recalculate token counts
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

DEFAULT_KEEP_RECENT = 6


class Compactor:
    """
    Stateless compaction engine.

    Instantiate once and call compact() whenever the auto-compact threshold
    is reached.
    """

    def __init__(
        self,
        provider,  # BaseProvider
        model: str,
        max_output_tokens: int = 8_000,
        keep_recent: int = DEFAULT_KEEP_RECENT,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._keep_recent = keep_recent

    async def compact(
        self,
        session: Session,
        system_prompt: str,
        partial: bool = True,
    ) -> CompactionResult:
        """Compact the session's message history."""
        messages = session.messages
        tokens_before = session.token_usage.total
        msg_count_before = len(messages)

        if not messages:
            return CompactionResult(
                summary="",
                messages_before=0,
                messages_after=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        # Choose compaction mode
        if partial and len(messages) > self._keep_recent:
            to_summarise = messages[: -self._keep_recent]
            to_keep = messages[-self._keep_recent :]
            prompt_template = PARTIAL_COMPACT_PROMPT
        else:
            to_summarise = messages
            to_keep = []
            prompt_template = BASE_COMPACT_PROMPT

        # Call the model
        summary_raw = await self._call_compact_model(to_summarise, prompt_template)

        # Parse summary
        summary = extract_summary(summary_raw)
        compact_message_text = build_compact_user_message(summary)

        # Rebuild message list
        compact_msg = Message(
            role=Role.USER,
            content=compact_message_text,
            token_count=count_tokens(compact_message_text, self._model),
            is_compact_summary=True,
        )

        new_messages: list[Message] = [compact_msg] + list(to_keep)

        # Update session
        new_total = _recalculate_tokens_for(session, new_messages, system_prompt)
        session.replace_messages(new_messages, new_total)

        return CompactionResult(
            summary=summary,
            messages_before=msg_count_before,
            messages_after=len(new_messages),
            tokens_before=tokens_before,
            tokens_after=new_total,
            kept_recent_count=len(to_keep),
        )

    async def _call_compact_model(
        self,
        messages_to_summarise: list[Message],
        prompt_template: str,
    ) -> str:
        """Send the compaction request to the model."""
        # Convert messages to simple user/assistant format for compaction
        # (strip tool messages to avoid confusing the compaction model)
        api_messages = []
        for m in messages_to_summarise:
            if m.role == Role.TOOL:
                # Convert tool results to user messages for compaction
                api_messages.append({
                    "role": "user",
                    "content": f"[Tool result for {m.tool_name or 'unknown'}]: {m.content}",
                })
            elif m.role == Role.ASSISTANT and m.tool_calls:
                # Include tool call info in content
                tc_info = ", ".join(tc.name for tc in m.tool_calls)
                text = m.content or ""
                api_messages.append({
                    "role": "assistant",
                    "content": f"{text}\n[Called tools: {tc_info}]" if text else f"[Called tools: {tc_info}]",
                })
            else:
                api_messages.append(m.to_openai_dict())

        api_messages.append({
            "role": "user",
            "content": (
                "Please create a detailed summary of this conversation following "
                "the instructions in the system prompt."
            ),
        })

        response = await self._provider.chat(
            messages=api_messages,
            system=prompt_template,
            max_tokens=self._max_output_tokens,
            temperature=0.0,
        )
        return response.content or ""


def _recalculate_tokens_for(
    session: Session,
    messages: list[Message],
    system_prompt: str,
) -> int:
    """Recalculate the total token count for a given message list + system prompt."""
    all_msgs = [{"role": "system", "content": system_prompt}]
    all_msgs += [m.to_openai_dict() for m in messages]
    total = count_messages_tokens(all_msgs, session.model)
    return total
