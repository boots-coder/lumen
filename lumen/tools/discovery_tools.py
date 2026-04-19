"""
Discovery / self-reflection utility tools.

As the tool catalogue grows (MCP servers, sub-agents, custom tools), the
model can lose track of what it has. Two small helpers close that gap:

  · `tool_search` — rank registered tools by relevance to a query so the
    model can discover what's available before a complex task.
  · `brief` — condensed "what's going on in this session so far" — recent
    messages + session-memory entries fed through the model for a short
    narrative. Useful before spawning a sub-agent, or when the user
    returns to a long session.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from ..agent import Agent


# ─────────────────────────────────────────────────────────────────────────────
# tool_search
# ─────────────────────────────────────────────────────────────────────────────

class ToolSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Natural-language description of what you want to do. "
            "e.g. 'read a file', 'search git log', 'query github issues'."
        ),
    )
    top_k: int = Field(
        default=8,
        description="How many results to return (1-20)",
    )


class ToolSearchTool(Tool[ToolSearchInput]):
    """Rank registered tools by relevance to a query string."""

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return (
            "Search the list of available tools for ones relevant to a task. "
            "Returns tool names + short descriptions ranked by relevance. "
            "Useful when the tool catalogue is large (e.g. with many MCP servers)."
        )

    @property
    def input_schema(self) -> type[ToolSearchInput]:
        return ToolSearchInput

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def execute(self, input_data: ToolSearchInput) -> ToolResult:
        query = input_data.query.strip()
        if not query:
            return ToolResult(
                success=False, output="", error="Empty query.",
            )
        k = max(1, min(20, input_data.top_k))
        query_tokens = _tokenize(query)

        registry = self._agent._tool_registry
        scored: list[tuple[float, str, str]] = []
        for tname in registry.list_tools():
            if tname == self.name:
                continue  # don't recommend ourselves
            tool = registry.get(tname)
            if tool is None:
                continue
            desc = tool.description
            score = _score(query_tokens, tname, desc)
            if score > 0:
                scored.append((score, tname, desc))

        if not scored:
            return ToolResult(
                success=True,
                output=f"No tools matched query {query!r}. "
                       f"Available: {', '.join(registry.list_tools())}",
            )

        scored.sort(key=lambda t: t[0], reverse=True)
        lines = [f"Top {min(k, len(scored))} tool(s) for {query!r}:"]
        for score, tname, desc in scored[:k]:
            short = desc.strip().split("\n", 1)[0][:140]
            lines.append(f"  · {tname}  —  {short}")
        return ToolResult(success=True, output="\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# brief
# ─────────────────────────────────────────────────────────────────────────────

class BriefInput(BaseModel):
    focus: str = Field(
        default="",
        description=(
            "Optional aspect to emphasize (e.g. 'open questions', "
            "'what files we touched', 'user preferences'). Empty = balanced summary."
        ),
    )


class BriefTool(Tool[BriefInput]):
    """Ask the provider for a condensed summary of the current session."""

    @property
    def name(self) -> str:
        return "brief"

    @property
    def description(self) -> str:
        return (
            "Produce a short narrative of what's happened in this session so far — "
            "recent messages + accumulated memory entries. Useful before spawning "
            "a sub-agent or when re-orienting in a long session."
        )

    @property
    def input_schema(self) -> type[BriefInput]:
        return BriefInput

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def execute(self, input_data: BriefInput) -> ToolResult:
        agent = self._agent
        recent = agent._session.messages[-12:]
        convo = "\n".join(
            f"[{m.role.value}] {(m.content or '')[:600]}"
            for m in recent
            if (m.content or "").strip()
        )

        mem_lines: list[str] = []
        if agent._session_memory is not None and len(agent._session_memory) > 0:
            for e in agent._session_memory._entries[-30:]:
                mem_lines.append(
                    f"  - [{e.category.value}] {e.content[:240]}"
                )
        memory_block = "\n".join(mem_lines) if mem_lines else "  (none)"

        focus_hint = (
            f"\nEmphasize: {input_data.focus.strip()}"
            if input_data.focus.strip() else ""
        )

        prompt = (
            "Summarize the current coding session for the user. 4-8 sentences. "
            "Cover: what they're working on, key decisions made, files touched, "
            "open threads. Be concrete — mention specific functions/files if "
            f"named. Skip pleasantries.{focus_hint}\n\n"
            f"## Recent messages\n{convo or '(none)'}\n\n"
            f"## Memory entries\n{memory_block}\n"
        )

        try:
            response = await agent._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="You write terse, information-dense summaries.",
                max_tokens=500,
                temperature=0.2,
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

        summary = (response.content or "").strip()
        if not summary:
            return ToolResult(
                success=False, output="", error="Empty response from provider.",
            )
        return ToolResult(success=True, output=summary)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _score(query_tokens: set[str], name: str, description: str) -> float:
    """Blend tool-name matches (strong) with description matches (weaker)."""
    if not query_tokens:
        return 0.0
    name_tokens = _tokenize(name)
    desc_tokens = _tokenize(description)
    name_hits = len(query_tokens & name_tokens)
    desc_hits = len(query_tokens & desc_tokens)
    # Substring boost: exact substring of the query inside tool name
    q_lower = " ".join(sorted(query_tokens))
    substr_boost = 0.0
    for qt in query_tokens:
        if qt in name.lower():
            substr_boost += 0.5
        elif qt in description.lower():
            substr_boost += 0.1
    return name_hits * 3.0 + desc_hits * 1.0 + substr_boost
