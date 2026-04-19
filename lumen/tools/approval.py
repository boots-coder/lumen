"""
Approval tool — pauses mid-response to request explicit user approval.

This is the core enforcement mechanism for Review Mode. When the AI calls
this tool, the agent loop blocks until the user approves or rejects via the
console, and the decision is returned as a tool result so the model knows
whether to proceed to the next phase, revise the current one, or skip ahead.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .base import Tool, ToolResult


# ── Phase + response types ───────────────────────────────────────────────────

class ApprovalPhase(str, Enum):
    """Review Mode phase being gated by the approval request."""
    DESIGN = "design"      # Phase 1: function/class signatures proposed
    PIPELINE = "pipeline"  # Phase 2: data flow diagram
    IMPL = "impl"          # Phase 3: after each function implementation


@dataclass
class ApprovalResponse:
    """User's decision on an approval request."""
    approved: bool
    feedback: str = ""                # free-form user feedback on rejection
    skip_remaining: bool = False      # user said "全部写完" / "skip"


# ── Handler protocol ─────────────────────────────────────────────────────────

class ApprovalHandler(Protocol):
    async def request(
        self,
        phase: ApprovalPhase,
        title: str,
        content: str,
    ) -> ApprovalResponse: ...


# ── Console handler ──────────────────────────────────────────────────────────

_PHASE_BORDER = {
    ApprovalPhase.DESIGN: "cyan",
    ApprovalPhase.PIPELINE: "blue",
    ApprovalPhase.IMPL: "yellow",
}

_PHASE_LABEL = {
    ApprovalPhase.DESIGN: "DESIGN",
    ApprovalPhase.PIPELINE: "PIPELINE",
    ApprovalPhase.IMPL: "IMPL",
}

_APPROVE_WORDS = {"", "y", "yes", "ok", "确认"}
_SKIP_WORDS = {"skip", "全部写完", "write all"}
_REJECT_WORDS = {"n", "no", "reject"}


def _truncate(content: str, head: int = 40, tail: int = 20, limit: int = 80) -> str:
    """If content exceeds `limit` lines, show head + elision + tail."""
    lines = content.splitlines()
    if len(lines) <= limit:
        return content
    elided = len(lines) - head - tail
    return "\n".join(
        lines[:head]
        + [f"... ({elided} lines elided) ..."]
        + lines[-tail:]
    )


class ConsoleApprovalHandler:
    """
    Default handler: renders a rich Panel with phase header + content,
    prompts user for y/n + optional feedback via prompt_toolkit.
    """

    def __init__(self, pt_session: Any = None, console: Any = None) -> None:
        self._pt_session = pt_session
        self._console = console

    async def request(
        self,
        phase: ApprovalPhase,
        title: str,
        content: str,
    ) -> ApprovalResponse:
        self._render_panel(phase, title, content)
        answer = await self._prompt(
            "回车=通过  n=驳回  skip=全部通过  (或直接输入反馈) › "
        )
        return await self._interpret(answer)

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render_panel(
        self,
        phase: ApprovalPhase,
        title: str,
        content: str,
    ) -> None:
        try:
            from rich.console import Console
            from rich.panel import Panel
        except ImportError:
            # Fallback: plain text
            console = self._console
            print(f"\n⚑ {_PHASE_LABEL[phase]} — {title}\n")
            print(_truncate(content))
            print()
            return

        console = self._console or Console()
        border = _PHASE_BORDER.get(phase, "white")
        badge = f"⚑ {_PHASE_LABEL[phase]}"
        panel_title = f"[{border}]{badge}[/{border}] — {title}"
        body = _truncate(content)
        console.print(Panel(body, title=panel_title, border_style=border))

    # ── Prompting ────────────────────────────────────────────────────────────

    async def _prompt(self, message: str) -> str:
        try:
            if self._pt_session is not None:
                # default="" + multiline=False → bare Enter submits empty string,
                # which _interpret() treats as approval.
                result = await self._pt_session.prompt_async(
                    message, default="", multiline=False,
                )
                return (result or "").strip()
            return (await asyncio.to_thread(input, message)).strip()
        except KeyboardInterrupt:
            return "__cancelled__"
        except EOFError:
            return "__cancelled__"

    async def _interpret(self, answer: str) -> ApprovalResponse:
        if answer == "__cancelled__":
            return ApprovalResponse(
                approved=False,
                feedback="user cancelled",
            )

        lowered = answer.strip().lower()

        if lowered in _APPROVE_WORDS:
            return ApprovalResponse(approved=True, skip_remaining=False)

        if lowered in _SKIP_WORDS:
            return ApprovalResponse(approved=True, skip_remaining=True)

        if lowered in _REJECT_WORDS:
            feedback = await self._prompt("反馈 › ")
            if feedback == "__cancelled__":
                feedback = "user cancelled"
            return ApprovalResponse(
                approved=False,
                feedback=feedback or "(no specific feedback)",
            )

        # Anything else starting with non-y → treat as inline feedback
        return ApprovalResponse(approved=False, feedback=answer)


# ── Tool input schema + tool ─────────────────────────────────────────────────

class ApprovalToolInput(BaseModel):
    phase: ApprovalPhase = Field(
        ...,
        description=(
            "Which Review Mode phase this approval gates: "
            "'design' (signatures), 'pipeline' (data flow), 'impl' (function body)."
        ),
    )
    title: str = Field(
        ...,
        description="Short 5-10 word header, e.g. 'Design for the config parser'",
    )
    summary: str = Field(
        ...,
        description=(
            "The content for user to review: signatures (design), "
            "data flow diagram (pipeline), or the just-written function body "
            "+ explanation (impl)"
        ),
    )


_DESCRIPTION = """Pause execution and request explicit user approval for the current Review Mode phase.

Call this tool ONLY when Review Mode is active. Use it to gate transitions:
- At end of Phase 1 (Design): phase="design", summary=the function/class signatures you propose
- At end of Phase 2 (Data Flow): phase="pipeline", summary=the data flow diagram
- After each function implementation in Phase 3: phase="impl", summary=the function body + how data transforms

The tool blocks until the user responds. Result will be one of:
- APPROVED — continue to next phase
- APPROVED + skip_remaining — user wants you to finish everything without further approvals
- REJECTED + feedback — revise the current phase per user feedback, then call request_approval again

Do NOT call this tool outside Review Mode. Do NOT call it for read-only tasks."""


class RequestApprovalTool(Tool[ApprovalToolInput]):
    """
    Tool that the model calls to pause and request user approval between
    Review Mode phases. The tool delegates to an ApprovalHandler which
    owns the actual UX (console panel, prompt, etc.).
    """

    @property
    def name(self) -> str:
        return "request_approval"

    @property
    def description(self) -> str:
        return _DESCRIPTION

    @property
    def input_schema(self) -> type[ApprovalToolInput]:
        return ApprovalToolInput

    def __init__(self, handler: ApprovalHandler) -> None:
        self._handler = handler

    async def execute(self, input_data: ApprovalToolInput) -> ToolResult:
        resp = await self._handler.request(
            input_data.phase,
            input_data.title,
            input_data.summary,
        )

        if resp.approved:
            if resp.skip_remaining:
                msg = (
                    "APPROVED. User requested to SKIP remaining phases — "
                    "complete all remaining work in one go without further approvals."
                )
            else:
                msg = "APPROVED. Proceed to the next phase or to the next function."
            return ToolResult(success=True, output=msg)

        feedback = resp.feedback or "(no specific feedback)"
        msg = (
            f"REJECTED. User feedback: {feedback}. "
            "Revise this phase and call request_approval again."
        )
        # success=True because the tool ran correctly; the REJECTED signal is in the content
        return ToolResult(success=True, output=msg)
