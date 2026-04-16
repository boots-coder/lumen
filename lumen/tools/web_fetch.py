"""
Web fetch tool — fetch and extract readable content from a URL.

Design:
- Uses httpx.AsyncClient for async HTTP requests
- Simple regex-based HTML text extraction (no heavy deps)
- Strips script, style, nav, footer tags and their content
- Handles redirects, timeouts, and common HTTP errors
- Truncates output to configurable max_length
"""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field

from .base import Tool, ToolResult

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; LumenAgent/1.0; +https://github.com/lumen-agent)"
)

# Regex patterns for HTML stripping
_STRIP_TAGS_WITH_CONTENT = re.compile(
    r"<\s*(script|style|nav|footer|header|aside|iframe|noscript)"
    r"[^>]*>.*?</\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_STRIP_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_STRIP_ALL_TAGS = re.compile(r"<[^>]+>")
_COLLAPSE_WHITESPACE = re.compile(r"[ \t]+")
_COLLAPSE_NEWLINES = re.compile(r"\n{3,}")


def _extract_text(html: str) -> str:
    """Extract readable text from HTML using regex-based stripping."""
    text = _STRIP_COMMENTS.sub("", html)
    text = _STRIP_TAGS_WITH_CONTENT.sub("", text)
    text = _STRIP_ALL_TAGS.sub("\n", text)
    text = _COLLAPSE_WHITESPACE.sub(" ", text)
    text = _COLLAPSE_NEWLINES.sub("\n\n", text)
    return text.strip()


class WebFetchInput(BaseModel):
    """Input schema for WebFetch tool."""

    url: str = Field(description="URL to fetch content from")
    extract_text: bool = Field(
        default=True,
        description="Extract readable text (strip HTML). Set False for raw content.",
    )
    max_length: int = Field(
        default=50000,
        description="Maximum content length in characters (default 50000)",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers to include in the request",
    )


class WebFetchTool(Tool[WebFetchInput]):
    """
    Fetch content from a URL.

    Can return raw HTML or extracted readable text.
    Uses httpx for async HTTP requests with redirect following.
    """

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch content from a URL. Can return raw HTML or extracted readable text.\n\n"
            "Parameters:\n"
            "  url          — URL to fetch (required)\n"
            "  extract_text — strip HTML and return readable text (default True)\n"
            "  max_length   — truncate content to this many characters (default 50000)\n"
            "  headers      — optional dict of HTTP headers\n\n"
            "Supports text/html, application/json, and text/plain content types.\n"
            "Follows up to 5 redirects automatically.\n\n"
            "Examples:\n"
            '  {"url": "https://example.com/article"}\n'
            '  {"url": "https://api.example.com/data.json", "extract_text": false}\n'
            '  {"url": "https://example.com", "headers": {"Authorization": "Bearer tok"}}'
        )

    @property
    def input_schema(self) -> type[WebFetchInput]:
        return WebFetchInput

    async def execute(self, input_data: WebFetchInput) -> ToolResult:
        try:
            # Validate URL
            url = input_data.url.strip()
            if not url.startswith(("http://", "https://")):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid URL scheme — must start with http:// or https://: {url}",
                )

            # Build headers
            request_headers = {"User-Agent": _DEFAULT_USER_AGENT}
            if input_data.headers:
                request_headers.update(input_data.headers)

            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                timeout=httpx.Timeout(10.0, connect=5.0),
            ) as client:
                response = await client.get(url, headers=request_headers)

            # Check for HTTP errors
            if response.status_code >= 400:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"HTTP {response.status_code}: {response.reason_phrase} for {url}",
                )

            content_type = response.headers.get("content-type", "")
            content = response.text

            # For JSON or plain text, skip HTML extraction
            is_html = "text/html" in content_type
            should_extract = input_data.extract_text and is_html

            if should_extract:
                content = _extract_text(content)

            # Truncate to max_length
            max_len = max(100, input_data.max_length)
            truncated = False
            if len(content) > max_len:
                content = content[:max_len]
                truncated = True

            header = (
                f"[web_fetch] url={url}  "
                f"status={response.status_code}  "
                f"content_type={content_type.split(';')[0].strip()}  "
                f"length={len(content)}"
                + ("  (truncated)" if truncated else "")
            )
            output = header + "\n\n" + content

            return ToolResult(success=True, output=output)

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error=f"Request timed out fetching: {input_data.url}",
            )
        except httpx.ConnectError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Connection failed (DNS or network error): {e}",
            )
        except httpx.TooManyRedirects:
            return ToolResult(
                success=False,
                output="",
                error=f"Too many redirects (>5) for: {input_data.url}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"web_fetch error: {e}")
