"""
Lifecycle hooks — pre_tool / post_tool execution hooks.

Mirrors src/services/tools/toolHooks.ts:
  - Pre-tool hooks: can block, modify input, or allow
  - Post-tool hooks: can modify output, add context, prevent continuation
  - Post-tool-failure hooks: can provide remediation context
  - Hooks are registered per-tool or globally
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

class HookBehavior(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PreToolHookResult:
    """Result from a pre-tool hook."""
    behavior: HookBehavior = HookBehavior.ALLOW
    message: str | None = None           # Message to show user
    blocking_error: str | None = None    # If set, tool is blocked
    updated_input: dict[str, Any] | None = None  # Modified tool input


@dataclass
class PostToolHookResult:
    """Result from a post-tool hook."""
    message: str | None = None           # Additional message
    prevent_continuation: bool = False   # Stop the tool loop
    additional_context: list[str] | None = None  # Extra context to inject


@dataclass
class PostToolFailureHookResult:
    """Result from a post-tool-failure hook."""
    message: str | None = None
    remediation: str | None = None       # Suggested fix


# Hook function signatures
PreToolHook = Callable[[str, dict[str, Any]], Any]
PostToolHook = Callable[[str, dict[str, Any], str, bool], Any]
PostToolFailureHook = Callable[[str, dict[str, Any], str], Any]


# ── Hook registry ────────────────────────────────────────────────────────────

@dataclass
class _HookEntry:
    """A registered hook with optional tool filter."""
    fn: Any
    tool_name: str | None = None  # None = matches all tools


class HookRegistry:
    """
    Registry for lifecycle hooks.

    Hooks are called in registration order. Multiple hooks per event
    are supported. A blocking pre-tool hook stops further hooks from running.
    """

    def __init__(self) -> None:
        self._pre_tool: list[_HookEntry] = []
        self._post_tool: list[_HookEntry] = []
        self._post_tool_failure: list[_HookEntry] = []

    # ── Registration ─────────────────────────────────────────────────────

    def on_pre_tool(
        self,
        hook: PreToolHook,
        tool_name: str | None = None,
    ) -> None:
        """
        Register a pre-tool hook.

        hook(tool_name, tool_input) → PreToolHookResult | None
        If tool_name is specified, only fires for that tool.
        """
        self._pre_tool.append(_HookEntry(fn=hook, tool_name=tool_name))

    def on_post_tool(
        self,
        hook: PostToolHook,
        tool_name: str | None = None,
    ) -> None:
        """
        Register a post-tool hook.

        hook(tool_name, tool_input, output, is_error) → PostToolHookResult | None
        """
        self._post_tool.append(_HookEntry(fn=hook, tool_name=tool_name))

    def on_post_tool_failure(
        self,
        hook: PostToolFailureHook,
        tool_name: str | None = None,
    ) -> None:
        """
        Register a post-tool-failure hook.

        hook(tool_name, tool_input, error_message) → PostToolFailureHookResult | None
        """
        self._post_tool_failure.append(_HookEntry(fn=hook, tool_name=tool_name))

    # ── Execution ────────────────────────────────────────────────────────

    async def run_pre_tool_hooks(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PreToolHookResult:
        """
        Run all matching pre-tool hooks.

        Returns the first blocking result, or ALLOW if none block.
        """
        for entry in self._pre_tool:
            if entry.tool_name and entry.tool_name != tool_name:
                continue

            try:
                result = entry.fn(tool_name, tool_input)
                if inspect.isawaitable(result):
                    result = await result

                if result is None:
                    continue

                if isinstance(result, PreToolHookResult):
                    if result.blocking_error:
                        logger.info(
                            "Pre-tool hook blocked %s: %s",
                            tool_name, result.blocking_error,
                        )
                        return result
                    if result.behavior == HookBehavior.DENY:
                        return result
                    if result.updated_input is not None:
                        # Return with modified input
                        return result
            except Exception as e:
                logger.error("Pre-tool hook error for %s: %s", tool_name, e)

        return PreToolHookResult(behavior=HookBehavior.ALLOW)

    async def run_post_tool_hooks(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        output: str,
        is_error: bool,
    ) -> PostToolHookResult:
        """Run all matching post-tool hooks."""
        combined = PostToolHookResult()

        for entry in self._post_tool:
            if entry.tool_name and entry.tool_name != tool_name:
                continue

            try:
                result = entry.fn(tool_name, tool_input, output, is_error)
                if inspect.isawaitable(result):
                    result = await result

                if result is None:
                    continue

                if isinstance(result, PostToolHookResult):
                    if result.prevent_continuation:
                        combined.prevent_continuation = True
                    if result.message:
                        combined.message = (
                            (combined.message or "") + "\n" + result.message
                        ).strip()
                    if result.additional_context:
                        if combined.additional_context is None:
                            combined.additional_context = []
                        combined.additional_context.extend(result.additional_context)
            except Exception as e:
                logger.error("Post-tool hook error for %s: %s", tool_name, e)

        return combined

    async def run_post_tool_failure_hooks(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        error_message: str,
    ) -> PostToolFailureHookResult:
        """Run all matching post-tool-failure hooks."""
        combined = PostToolFailureHookResult()

        for entry in self._post_tool_failure:
            if entry.tool_name and entry.tool_name != tool_name:
                continue

            try:
                result = entry.fn(tool_name, tool_input, error_message)
                if inspect.isawaitable(result):
                    result = await result

                if result is None:
                    continue

                if isinstance(result, PostToolFailureHookResult):
                    if result.remediation:
                        combined.remediation = (
                            (combined.remediation or "") + "\n" + result.remediation
                        ).strip()
                    if result.message:
                        combined.message = (
                            (combined.message or "") + "\n" + result.message
                        ).strip()
            except Exception as e:
                logger.error("Post-tool-failure hook error for %s: %s", tool_name, e)

        return combined

    # ── Info ─────────────────────────────────────────────────────────────

    @property
    def has_hooks(self) -> bool:
        return bool(self._pre_tool or self._post_tool or self._post_tool_failure)

    def clear(self) -> None:
        """Remove all registered hooks."""
        self._pre_tool.clear()
        self._post_tool.clear()
        self._post_tool_failure.clear()
