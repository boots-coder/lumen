"""
Agent — the single public class users interact with.

Usage:
    from engram import Agent

    agent = Agent(api_key="sk-...", model="gpt-4o")
    response = await agent.chat("Explain this codebase")

    async for chunk in agent.stream("Refactor the auth module"):
        print(chunk, end="", flush=True)

Everything else (memory, git state, token counting, auto-compaction) happens
automatically. Users never touch Sessions, Compactors, or Providers directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from ._types import ChatResponse, CompactionResult, TokenUsage, ToolCall
from .compact.auto_compact import (
    MAX_CONSECUTIVE_FAILURES,
    assess_context_window,
)
from .compact.compactor import Compactor
from .context.session import Session
from .context.system_prompt import SystemPromptBuilder
from .providers.factory import create_provider
from .tokens.counter import get_context_window, get_max_output_tokens
from .tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """
    A stateful LLM agent with automatic context management.

    Parameters
    ----------
    api_key : str
        API key for the model provider.
    model : str
        Model identifier, e.g. "gpt-4o", "claude-sonnet-4-6", "llama3.1".
    base_url : str | None
        API base URL. Auto-detected from model name if None.
        Examples:
          - OpenAI  : "https://api.openai.com/v1"
          - Anthropic: "https://api.anthropic.com"
          - Ollama  : "http://localhost:11434/v1"
          - Azure   : "https://<resource>.openai.azure.com/openai/deployments/<deployment>"
    system_prompt : str | None
        Additional system instructions appended after Engram's base prompt.
    context_window : int | None
        Override the model's context window (tokens). Auto-detected if None.
    max_output_tokens : int | None
        Override the model's max output tokens. Auto-detected if None.
    cwd : Path | str | None
        Working directory for ENGRAM.md discovery and git state.
        Defaults to the current working directory.
    inject_git_state : bool
        Inject a git status snapshot into the system prompt. Default True.
    inject_memory : bool
        Discover and load ENGRAM.md memory files. Default True.
    auto_compact : bool
        Automatically compact the context when approaching the window limit.
        Default True.
    keep_recent : int
        Number of recent messages to preserve verbatim during partial compaction.
        Default 6.
    compact_model : str | None
        Use a different (stronger) model for compaction summarisation.
        Falls back to the main model if None.
    compact_api_key : str | None
        API key for the compact model (if different from the main key).
    compact_base_url : str | None
        Base URL for the compact model (if different).
    temperature : float
        Sampling temperature for chat/stream calls. Default 0.7.
    timeout : float
        HTTP timeout in seconds. Default 120.
    session : Session | None
        Restore a previously saved session. See Agent.load_session().
    tools : list[Tool] | None
        List of tools to make available to the agent (e.g., FileReadTool, GlobTool).
        If None, no tools are available.
    max_tool_calls : int
        Maximum number of tool calls allowed per chat turn (防止无限循环). Default 20.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        *,
        system_prompt: str | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        cwd: Path | str | None = None,
        inject_git_state: bool = True,
        inject_memory: bool = True,
        auto_compact: bool = True,
        keep_recent: int = 6,
        compact_model: str | None = None,
        compact_api_key: str | None = None,
        compact_base_url: str | None = None,
        temperature: float = 0.7,
        timeout: float = 120.0,
        session: Session | None = None,
        tools: list[Tool] | None = None,
        max_tool_calls: int = 20,
        code_reading_mode: bool = False,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._auto_compact = auto_compact
        self._max_tool_calls = max_tool_calls

        # Resolved limits
        self._context_window = context_window or get_context_window(model)
        self._max_output_tokens = max_output_tokens or get_max_output_tokens(model)

        # Tool registry
        self._tool_registry = ToolRegistry()
        if tools:
            for tool in tools:
                self._tool_registry.register(tool)
            logger.debug(f"Registered {len(tools)} tools: {self._tool_registry.list_tools()}")

        # Working directory
        self._cwd = Path(cwd).resolve() if cwd else Path.cwd()

        # Provider
        self._provider, self._base_url = create_provider(
            api_key, model, base_url, timeout
        )

        # Compact provider (may differ from main provider)
        if compact_model or compact_api_key or compact_base_url:
            _cmodel = compact_model or model
            _ckey = compact_api_key or api_key
            _curl = compact_base_url or base_url
            self._compact_provider, _ = create_provider(_ckey, _cmodel, _curl, timeout)
            _compact_model_name = _cmodel
        else:
            self._compact_provider = self._provider
            _compact_model_name = model

        # System prompt builder (lazy — built on first chat call)
        self._prompt_builder = SystemPromptBuilder(
            extra_instructions=system_prompt or "",
            cwd=self._cwd,
            inject_git_state=inject_git_state,
            inject_memory=inject_memory,
            code_reading_mode=code_reading_mode,
        )

        # Session (restore or create fresh)
        self._session = session or Session(model=model)
        self._session.token_usage.context_window = self._context_window

        # Compactor
        self._compactor = Compactor(
            provider=self._compact_provider,
            model=_compact_model_name,
            max_output_tokens=min(self._max_output_tokens, 8_000),
            keep_recent=keep_recent,
        )

        logger.debug(
            "Agent ready: model=%s context_window=%d max_output=%d auto_compact=%s",
            model, self._context_window, self._max_output_tokens, auto_compact,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        on_tool_call: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> ChatResponse:
        """
        Send a message and return the full response.

        Context management (memory injection, token counting, auto-compaction)
        and tool calling are handled transparently.

        Args:
            on_tool_call: Optional async/sync callable(name, arguments) fired
                          before each tool is executed. Useful for UI progress.
            on_tool_result: Optional async/sync callable(name, output, is_error)
                            fired after each tool completes.

        Returns:
            ChatResponse with .content (str) and .token_usage.
        """
        import inspect

        async def _fire(fn, *args):
            if fn is None:
                return
            result = fn(*args)
            if inspect.isawaitable(result):
                await result

        system = await self._prompt_builder.build()

        # Check if we need to compact BEFORE adding the new message
        compaction_result: CompactionResult | None = None
        if self._auto_compact:
            compaction_result = await self._maybe_compact(system)

        # Add user message to session
        self._session.add_user(message)

        # Tool calling loop
        tool_call_count = 0
        final_content = ""

        while tool_call_count < self._max_tool_calls:
            # Prepare tools schema if tools are registered
            tools_schema = None
            if len(self._tool_registry) > 0:
                # Determine format based on model
                if any(x in self._model.lower() for x in ["gpt", "o1", "o3", "deepseek"]):
                    tools_schema = self._tool_registry.to_openai_tools()
                else:
                    tools_schema = self._tool_registry.to_anthropic_tools()

            # Call provider
            response = await self._provider.chat(
                messages=self._session.as_api_messages(),
                system=system,
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
                tools=tools_schema,
            )

            # Update token counts
            self._session.update_token_counts_from_api(
                response.prompt_tokens,
                response.completion_tokens,
                self._context_window,
            )

            # If there's text content, save it
            if response.content:
                final_content = response.content
                # Add assistant message to session
                self._session.add_assistant(response.content, response.completion_tokens)

            # If no tool calls, we're done
            if not response.tool_calls:
                break

            # Execute tool calls
            tool_call_count += 1
            logger.debug(f"Tool call round {tool_call_count}/{self._max_tool_calls}")
            await self._execute_tool_calls(
                response.tool_calls,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )

            # Continue loop to get next response

        # Check if we hit the limit
        if tool_call_count >= self._max_tool_calls:
            logger.warning(
                f"Reached maximum tool call limit ({self._max_tool_calls}). "
                "Stopping tool execution loop."
            )

        # Check thresholds for logging / warnings
        assess_context_window(
            self._session.token_usage.total,
            self._model,
            self._context_window,
            self._max_output_tokens,
        )

        return ChatResponse(
            content=final_content,
            token_usage=self._session.token_usage,
            was_compacted=compaction_result is not None,
            compaction_result=compaction_result,
        )

    async def stream(self, message: str) -> AsyncIterator[str]:
        """
        Stream the response, yielding text chunks as they arrive.

        Context management runs before streaming begins (same as chat()).
        Token counts are estimated from the streamed output since streaming
        APIs typically don't return token counts mid-stream.
        """
        system = await self._prompt_builder.build()

        # Pre-stream compaction check
        if self._auto_compact:
            await self._maybe_compact(system)

        self._session.add_user(message)

        collected: list[str] = []
        async for chunk in self._provider.stream(
            messages=self._session.as_api_messages(),
            system=system,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        ):
            collected.append(chunk)
            yield chunk

        full_text = "".join(collected)
        self._session.add_assistant(full_text)

        # Recalculate total including system prompt so the displayed number is
        # accurate (chat() gets real numbers from the API; stream() uses tiktoken).
        self._session.recalculate_tokens(system)

        # Re-assess after streaming completes
        assess_context_window(
            self._session.token_usage.total,
            self._model,
            self._context_window,
            self._max_output_tokens,
        )

    # ── Session management ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the conversation history, starting a fresh session."""
        self._session = Session(model=self._model)
        self._session.token_usage.context_window = self._context_window
        logger.debug("Session reset.")

    def save_session(self, path: str | Path) -> None:
        """Persist the current session to a JSON file."""
        self._session.save(path)

    @classmethod
    def load_session(
        cls,
        path: str | Path,
        api_key: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> "Agent":
        """
        Restore an Agent from a previously saved session file.

        Example:
            agent = Agent.load_session("session.json", api_key="sk-...", model="gpt-4o")
        """
        session = Session.load(path)
        _model = model or session.model
        return cls(api_key=api_key, model=_model, session=session, **kwargs)

    # ── Mode switching ────────────────────────────────────────────────────────

    def enable_code_reading_mode(self) -> None:
        """
        Activate Code Reading Mode.

        Adds a deep code archaeology layer on top of the base prompt.
        The model will: read before answering, cite file:line, follow call chains,
        explain WHY not just WHAT.  General capabilities are fully preserved.
        """
        self._prompt_builder.enable_code_reading_mode()
        logger.info("Code Reading Mode enabled.")

    def disable_code_reading_mode(self) -> None:
        """Deactivate Code Reading Mode and return to standard mode."""
        self._prompt_builder.disable_code_reading_mode()
        logger.info("Code Reading Mode disabled.")

    @property
    def code_reading_mode(self) -> bool:
        """Whether Code Reading Mode is currently active."""
        return self._prompt_builder.code_reading_mode

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def token_usage(self) -> TokenUsage:
        """Current token usage statistics."""
        return self._session.token_usage

    @property
    def messages(self) -> list:
        """Raw message list (read-only copy)."""
        return list(self._session.messages)

    @property
    def session(self) -> Session:
        """Direct access to the underlying Session (advanced use)."""
        return self._session

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        usage = self._session.token_usage
        return (
            f"Agent(model={self._model!r}, "
            f"tokens={usage.total}/{self._context_window}, "
            f"messages={len(self._session.messages)}, "
            f"compactions={self._session.compaction_count})"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _maybe_compact(self, system: str) -> CompactionResult | None:
        """
        Check whether auto-compaction should run and execute it if so.

        Implements the circuit-breaker: after MAX_CONSECUTIVE_FAILURES
        consecutive failures, stop trying.
        """
        if self._session.consecutive_compact_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Compaction circuit breaker open after %d consecutive failures. "
                "Skipping auto-compact.",
                MAX_CONSECUTIVE_FAILURES,
            )
            return None

        state = assess_context_window(
            self._session.token_usage.total,
            self._model,
            self._context_window,
            self._max_output_tokens,
        )

        if not state.should_compact:
            return None

        logger.info(
            "Auto-compact triggered at %d/%d tokens (%.1f%%).",
            state.total_tokens, state.context_window, state.percent_used,
        )

        try:
            result = await self._compactor.compact(
                session=self._session,
                system_prompt=system,
                partial=True,
            )
            logger.info(
                "Auto-compact succeeded: %d→%d tokens, %d→%d messages.",
                result.tokens_before, result.tokens_after,
                result.messages_before, result.messages_after,
            )
            return result
        except Exception as e:
            self._session.consecutive_compact_failures += 1
            logger.error(
                "Auto-compact failed (attempt %d/%d): %s",
                self._session.consecutive_compact_failures,
                MAX_CONSECUTIVE_FAILURES,
                e,
            )
            return None

    async def compact(self, partial: bool = True) -> CompactionResult:
        """
        Manually trigger context compaction.

        Args:
            partial: Keep recent messages verbatim if True (recommended).
        """
        system = await self._prompt_builder.build()
        result = await self._compactor.compact(
            session=self._session,
            system_prompt=system,
            partial=partial,
        )
        return result

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        on_tool_call: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> None:
        """
        Execute a list of tool calls and add results to the session.
        Fires on_tool_call / on_tool_result callbacks for UI feedback.
        """
        import inspect

        async def _fire(fn, *args):
            if fn is None:
                return
            r = fn(*args)
            if inspect.isawaitable(r):
                await r

        for tool_call in tool_calls:
            tool = self._tool_registry.get(tool_call.name)

            if not tool:
                logger.error(f"Tool not found: {tool_call.name}")
                await _fire(on_tool_call, tool_call.name, tool_call.arguments)
                self._session.add_tool_result(
                    tool_use_id=tool_call.id,
                    content=f"Error: Tool '{tool_call.name}' not found",
                    is_error=True,
                )
                await _fire(on_tool_result, tool_call.name, f"Tool not found", True)
                continue

            await _fire(on_tool_call, tool_call.name, tool_call.arguments)

            try:
                input_model = tool.input_schema(**tool_call.arguments)
                logger.debug(f"Executing tool: {tool_call.name}")
                result = await tool.execute(input_model)

                if result.success:
                    logger.debug(f"Tool {tool_call.name} succeeded")
                    self._session.add_tool_result(
                        tool_use_id=tool_call.id,
                        content=result.output,
                        is_error=False,
                    )
                    await _fire(on_tool_result, tool_call.name, result.output, False)
                else:
                    logger.warning(f"Tool {tool_call.name} failed: {result.error}")
                    error_msg = result.error or "Tool execution failed"
                    self._session.add_tool_result(
                        tool_use_id=tool_call.id,
                        content=f"Error: {error_msg}",
                        is_error=True,
                    )
                    await _fire(on_tool_result, tool_call.name, error_msg, True)

            except Exception as e:
                logger.error(f"Tool {tool_call.name} threw exception: {e}", exc_info=True)
                self._session.add_tool_result(
                    tool_use_id=tool_call.id,
                    content=f"Error: {str(e)}",
                    is_error=True,
                )
                await _fire(on_tool_result, tool_call.name, str(e), True)
