"""
Web search tool — search the web for information.

Design:
- Primary backend: duckduckgo_search library (no API key needed)
- Fallback: helpful error if library not installed
- Rate limit aware with configurable timeout
- Returns formatted results with titles, URLs, and snippets
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

try:
    from duckduckgo_search import DDGS

    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False


class WebSearchInput(BaseModel):
    """Input schema for WebSearch tool."""

    query: str = Field(description="Search query string")
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return (default 5, max 20)",
    )


class WebSearchTool(Tool[WebSearchInput]):
    """
    Search the web for information using DuckDuckGo.

    Returns titles, URLs, and text snippets for each result.
    Requires the optional `duckduckgo_search` package.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Returns titles, URLs, and snippets.\n\n"
            "Uses DuckDuckGo search (no API key required).\n\n"
            "Parameters:\n"
            "  query       — search query string (required)\n"
            "  max_results — maximum results to return (default 5, max 20)\n\n"
            "Examples:\n"
            '  {"query": "Python asyncio best practices"}\n'
            '  {"query": "httpx vs requests performance", "max_results": 10}'
        )

    @property
    def input_schema(self) -> type[WebSearchInput]:
        return WebSearchInput

    async def execute(self, input_data: WebSearchInput) -> ToolResult:
        try:
            if not _HAS_DDGS:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "The 'duckduckgo_search' package is not installed.\n"
                        "Install it with: pip install duckduckgo_search"
                    ),
                )

            max_results = min(max(1, input_data.max_results), 20)

            try:
                ddgs = DDGS()
                results = list(ddgs.text(
                    input_data.query,
                    max_results=max_results,
                ))
            except Exception as e:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Search request failed: {e}",
                )

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No results found for: {input_data.query}",
                )

            # Format results
            parts: list[str] = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("href", r.get("link", ""))
                snippet = r.get("body", r.get("snippet", ""))
                entry = f"{i}. [{title}]({url})"
                if snippet:
                    entry += f"\n   {snippet}"
                parts.append(entry)

            header = (
                f"[web_search] query={input_data.query!r}  "
                f"results={len(results)}"
            )
            output = header + "\n\n" + "\n\n".join(parts)

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output="", error=f"web_search error: {e}")
