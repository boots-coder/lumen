"""
Agent — the single public class users interact with.

Usage:
    from lumen import Agent

    agent = Agent(api_key="sk-...", model="gpt-4o")
    response = await agent.chat("Explain this codebase")

    async for chunk in agent.stream("Refactor the auth module"):
        print(chunk, end="", flush=True)

Production features (mirrors Claude Code):
  - API retry with exponential backoff + 529 fallback
  - Concurrent tool execution (read parallel, write serial)
  - max_tokens overflow auto-shrink
  - Tool result truncation
  - File state LRU cache
  - Permission system with safety guardrails
  - pre_tool / post_tool lifecycle hooks
  - Abort/cancellation support
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from ._types import ChatResponse, CompactionResult, TokenUsage, ToolCall
from .compact.auto_compact import (
    MAX_CONSECUTIVE_FAILURES,
    assess_context_window,
)
from .compact.compactor import Compactor
from .context.session import Session
from .context.session_memory import SessionMemoryManager
from .context.system_prompt import SystemPromptBuilder
from .context.transcript import TranscriptWriter
from .providers.factory import create_provider
from .tokens.counter import get_context_window, get_max_output_tokens
from .tools import Tool, ToolRegistry
from .providers.model_profiles import ModelProfile, detect_profile

# Services
from .services.abort import AbortController, AbortError
from .services.errors import classify_error, ErrorType
from .services.hooks import (
    HookBehavior, HookRegistry,
    PreToolHookResult, PostToolHookResult, PostToolFailureHookResult,
)
from .services.permissions import (
    PermissionBehavior, PermissionChecker, PermissionResult, PermissionRule,
)
from .services.retry import RetryConfig, with_retry
from .services.persistent_retry import PersistentRetryConfig, PersistentRetryManager
from .services.thinking import ThinkingConfig, ThinkingManager, ThinkingMode, ThinkingResult
from .services.prompt_cache import PromptCacheManager, CacheStats
from .services.structured_output import (
    OutputFormat, StructuredOutputConfig, StructuredResult,
    prepare_structured_request, validate_output,
)
from .services.review_state import ReviewState, ReviewPhase
from .services.subagent import SubAgentManager
from .services.tool_executor import (
    ToolExecutor, ToolExecRequest, ToolExecResult, ToolStatus,
)

# Utils
from .utils.file_state_cache import FileStateCache
from .utils.tool_result_truncation import truncate_tool_result

logger = logging.getLogger(__name__)


async def _fire(fn, *args):
    """Fire a callback, handling both sync and async."""
    if fn is None:
        return
    r = fn(*args)
    if inspect.isawaitable(r):
        await r


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
    system_prompt : str | None
        Additional system instructions appended after the base prompt.
    context_window : int | None
        Override the model's context window (tokens). Auto-detected if None.
    max_output_tokens : int | None
        Override the model's max output tokens. Auto-detected if None.
    cwd : Path | str | None
        Working directory for memory discovery and git state.
    inject_git_state : bool
        Inject a git status snapshot into context. Default True.
    inject_memory : bool
        Discover and load ENGRAM.md memory files. Default True.
    auto_compact : bool
        Automatically compact context when approaching window limit.
    keep_recent : int
        Number of recent messages to preserve during partial compaction.
    compact_model : str | None
        Use a different model for compaction summarisation.
    compact_api_key : str | None
        API key for the compact model (if different from the main key).
    compact_base_url : str | None
        Base URL for the compact model (if different).
    temperature : float
        Sampling temperature. Default 0.7.
    timeout : float
        HTTP timeout in seconds. Default 120.
    session : Session | None
        Restore a previously saved session.
    tools : list[Tool] | None
        List of tools to make available to the agent.
    max_tool_calls : int
        Maximum number of tool call rounds per chat turn. Default 20.
    permission_checker : PermissionChecker | None
        Permission system for tool safety guardrails.
    hooks : HookRegistry | None
        Lifecycle hooks for pre/post tool execution.
    retry_config : RetryConfig | None
        API retry configuration.
    fallback_model : str | None
        Model to fallback to on 529 overload.
    enable_file_cache : bool
        Enable file state LRU cache. Default True.
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
        review_mode: bool = False,
        # ── New production features ──────────────────────────────
        permission_checker: PermissionChecker | None = None,
        hooks: HookRegistry | None = None,
        retry_config: RetryConfig | None = None,
        fallback_model: str | None = None,
        enable_file_cache: bool = True,
        structured_output: StructuredOutputConfig | None = None,
        session_memory: bool = True,
        thinking: ThinkingConfig | None = None,
        persistent_retry: PersistentRetryConfig | None = None,
        auto_save: bool = True,
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

        # Detect provider type and model profile for format adaptation
        self._provider_type = self._detect_provider_type(model, base_url)
        self._model_profile = detect_profile(model, base_url)

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
            review_mode=review_mode,
        )

        # Session (restore or create fresh)
        self._session = session or Session(model=model)
        self._session.token_usage.context_window = self._context_window

        # Track whether context reminder has been injected
        self._context_injected = False

        # Compactor
        self._compactor = Compactor(
            provider=self._compact_provider,
            model=_compact_model_name,
            max_output_tokens=min(self._max_output_tokens, 8_000),
            keep_recent=keep_recent,
        )

        # ── Production features ──────────────────────────────────────────

        # Permission system
        self._permissions = permission_checker or PermissionChecker()

        # Lifecycle hooks
        self._hooks = hooks or HookRegistry()

        # Retry config
        self._retry_config = retry_config or RetryConfig(
            fallback_model=fallback_model,
        )

        # Abort controller (per-session)
        self._abort = AbortController()

        # File state cache
        self._file_cache = FileStateCache() if enable_file_cache else None

        # Prompt cache manager
        self._cache_manager = PromptCacheManager()

        # Structured output
        self._structured_output = structured_output

        # Session memory (dynamic key-fact extraction)
        self._session_memory_enabled = session_memory
        if session_memory:
            memory_path = self._cwd / ".engram" / "session_memory.json"
            self._session_memory = SessionMemoryManager(storage_path=memory_path)
            self._session_memory.load()
        else:
            self._session_memory = None

        # Thinking / extended reasoning
        self._thinking = thinking

        # Persistent retry (CI/batch mode)
        self._persistent_retry = persistent_retry
        self._persistent_retry_manager = (
            PersistentRetryManager() if persistent_retry and persistent_retry.enabled
            else None
        )

        # Sub-agent manager (not auto-registered — call enable_subagents())
        self._subagent_manager = SubAgentManager(self)

        # Review Mode state machine + approval tool (lazy-registered)
        self._review_state = ReviewState()
        self._approval_tool_registered = False

        # Tool executor (concurrent read, serial write)
        self._tool_executor = ToolExecutor(
            execute_fn=self._execute_single_tool,
            abort_event=self._abort.event,
        )

        # Transcript auto-save
        self._auto_save = auto_save
        if auto_save:
            self._transcript: TranscriptWriter | None = TranscriptWriter(
                session_id=self._session.session_id,
                cwd=self._cwd,
                model=model,
            )
        else:
            self._transcript = None

        logger.debug(
            "Agent ready: model=%s ctx=%d max_out=%d compact=%s provider=%s "
            "permissions=%s hooks=%s retry=%d file_cache=%s",
            model, self._context_window, self._max_output_tokens,
            auto_compact, self._provider_type,
            "on", "on" if self._hooks.has_hooks else "off",
            self._retry_config.max_retries,
            "on" if enable_file_cache else "off",
        )

    @staticmethod
    def _detect_provider_type(model: str, base_url: str | None) -> str:
        """Detect whether to use OpenAI or Anthropic message format."""
        if base_url and "anthropic.com" in base_url:
            return "anthropic"
        if model.startswith("claude"):
            return "anthropic"
        return "openai"

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        on_tool_call: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> ChatResponse:
        """
        Send a message and return the full response.

        Context management (memory injection, token counting, auto-compaction),
        tool calling, retry, permissions, and hooks are handled transparently.
        """
        system = await self._prompt_builder.build()

        # Check if we need to compact BEFORE adding the new message
        compaction_result: CompactionResult | None = None
        if self._auto_compact:
            compaction_result = await self._maybe_compact(system)

        # Inject context reminder on first turn (git, memory, cwd)
        if not self._context_injected:
            context_reminder = await self._prompt_builder.build_context_reminder()
            # Append session memory context to the first-turn reminder
            if self._session_memory and len(self._session_memory) > 0:
                memory_context = self._session_memory.get_relevant_context(
                    message, top_k=10,
                )
                if memory_context:
                    context_reminder = (
                        (context_reminder or "") + "\n\n" + memory_context
                    ).strip()
            if context_reminder:
                self._session.add_user(context_reminder)
            self._context_injected = True

        # Review Mode: auto-start state machine on new task
        if self._prompt_builder.review_mode:
            if self._review_state.phase in (ReviewPhase.IDLE, ReviewPhase.COMPLETE):
                self._review_state.start(task_summary=message[:200])

        # Prepend review-state system-reminder to user message (ephemeral per-turn)
        effective_message = message
        if self._prompt_builder.review_mode:
            state_reminder = self._review_state.as_system_reminder()
            if state_reminder:
                effective_message = f"{state_reminder}\n\n{message}"

        # Add user message to session
        user_msg = self._session.add_user(effective_message)

        # Auto-save: append user message to transcript
        if self._transcript:
            self._transcript.append_message(user_msg)

        # Tool calling loop
        tool_call_round = 0
        final_content = ""

        while tool_call_round < self._max_tool_calls:
            # Check abort
            if self._abort.is_aborted:
                break

            # Prepare tools schema — auto-adapted to model's pretrained format
            tools_schema = None
            if len(self._tool_registry) > 0:
                tools_schema = self._tool_registry.to_tools_for_profile(self._model_profile)

            # Call provider with retry
            response = await self._call_with_retry(
                system=system,
                tools_schema=tools_schema,
            )

            # Update token counts
            self._session.update_token_counts_from_api(
                response.prompt_tokens,
                response.completion_tokens,
                self._context_window,
            )

            # If no tool calls, save final content and break
            if not response.tool_calls:
                if response.content:
                    final_content = response.content
                    asst_msg = self._session.add_assistant(response.content, response.completion_tokens)
                    if self._transcript:
                        self._transcript.append_message(asst_msg)
                break

            # Assistant message WITH tool calls — must be in history
            asst_tc_msg = self._session.add_assistant_with_tool_calls(
                content=response.content,
                tool_calls=response.tool_calls,
                completion_tokens=response.completion_tokens,
            )
            if self._transcript:
                self._transcript.append_message(asst_tc_msg)

            if response.content:
                final_content = response.content

            # Execute tool calls with permissions, hooks, concurrency
            tool_call_round += 1
            logger.debug(f"Tool call round {tool_call_round}/{self._max_tool_calls}")
            await self._execute_tool_calls(
                response.tool_calls,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )

        # Check if we hit the limit
        if tool_call_round >= self._max_tool_calls:
            logger.warning(
                f"Reached maximum tool call limit ({self._max_tool_calls}). "
                "Stopping tool execution loop."
            )

        # Assess context window
        assess_context_window(
            self._session.token_usage.total,
            self._model,
            self._context_window,
            self._max_output_tokens,
        )

        # Session memory extraction (every 3 turns to avoid overhead)
        if (
            self._session_memory is not None
            and self._session_memory.turn_count % 3 == 0
            and len(self._session.messages) >= 2
        ):
            try:
                await self._session_memory.extract_from_turn(
                    self._session.messages, self._provider, self._model,
                )
                self._session_memory.save()
            except Exception as e:
                logger.debug("Session memory extraction skipped: %s", e)

        return ChatResponse(
            content=final_content,
            token_usage=self._session.token_usage,
            was_compacted=compaction_result is not None,
            compaction_result=compaction_result,
        )

    async def query(
        self,
        prompt: str,
        schema: type | dict[str, Any] | None = None,
        **chat_kwargs: Any,
    ) -> Any:
        """
        Convenience method: send *prompt* and return structured output.

        Temporarily activates structured output for this single call,
        then restores the previous config.

        Parameters
        ----------
        prompt : str
            The user prompt.
        schema : type[BaseModel] | dict | None
            The desired output schema.  A Pydantic model class or a raw
            JSON-schema dict.
        **chat_kwargs
            Forwarded to :meth:`chat`.

        Returns
        -------
        BaseModel | dict
            The parsed (and optionally validated) data.

        Raises
        ------
        ValueError
            If the model output cannot be parsed as valid JSON or fails
            schema validation.
        """
        config = StructuredOutputConfig(
            format=OutputFormat.JSON_SCHEMA if schema else OutputFormat.JSON,
            schema=schema,
        )
        previous = self._structured_output
        self._structured_output = config
        try:
            response = await self.chat(prompt, **chat_kwargs)
        finally:
            self._structured_output = previous

        result = validate_output(response.content, config)

        if not result.is_valid:
            raise ValueError(
                f"Structured output validation failed: {result.validation_errors}"
            )
        return result.data

    async def stream(self, message: str) -> AsyncIterator[str]:
        """Stream the response, yielding text chunks as they arrive."""
        system = await self._prompt_builder.build()

        if self._auto_compact:
            await self._maybe_compact(system)

        if not self._context_injected:
            context_reminder = await self._prompt_builder.build_context_reminder()
            if context_reminder:
                self._session.add_user(context_reminder)
            self._context_injected = True

        self._session.add_user(message)

        collected: list[str] = []
        async for chunk in self._provider.stream(
            messages=self._session.as_api_messages(self._provider_type),
            system=system,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
        ):
            collected.append(chunk)
            yield chunk

        full_text = "".join(collected)
        self._session.add_assistant(full_text)
        self._session.recalculate_tokens(system)

        assess_context_window(
            self._session.token_usage.total,
            self._model,
            self._context_window,
            self._max_output_tokens,
        )

    # ── Abort ─────────────────────────────────────────────────────────────────

    def abort(self, reason: str = "User cancelled") -> None:
        """Abort the current operation."""
        self._abort.abort(reason)

    def reset_abort(self) -> None:
        """Reset the abort controller for the next operation."""
        self._abort.reset()

    @property
    def is_aborted(self) -> bool:
        return self._abort.is_aborted

    # ── Session management ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the conversation history, starting a fresh session."""
        self._session = Session(model=self._model)
        self._session.token_usage.context_window = self._context_window
        self._context_injected = False
        self._prompt_builder.invalidate()
        self._abort.reset()
        self._review_state.reset()
        if self._file_cache:
            self._file_cache.clear()
        # Cleanup sub-agents on reset
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._subagent_manager.cleanup())
        except RuntimeError:
            pass
        logger.debug("Session reset.")

    async def close(self) -> None:
        """Cleanup all resources including transcript and background sub-agents."""
        if self._transcript:
            await self._transcript.flush()
            self._transcript.close()
        await self._subagent_manager.cleanup()
        logger.debug("Agent closed.")

    def enable_subagents(self) -> None:
        """Register the SubAgentTool, allowing the model to spawn child agents."""
        from .tools.subagent_tool import SubAgentTool
        tool = SubAgentTool(agent_manager=self._subagent_manager)
        self._tool_registry.register(tool)
        logger.debug("Sub-agent tool enabled.")

    @property
    def subagent_manager(self) -> SubAgentManager:
        """Access the sub-agent manager for programmatic spawning."""
        return self._subagent_manager

    def save_session(self, path: str | Path) -> None:
        self._session.save(path)

    @classmethod
    def load_session(
        cls,
        path: str | Path,
        api_key: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> "Agent":
        session = Session.load(path)
        _model = model or session.model
        return cls(api_key=api_key, model=_model, session=session, **kwargs)

    # ── Permission & Hook access ──────────────────────────────────────────────

    @property
    def permissions(self) -> PermissionChecker:
        return self._permissions

    @property
    def hooks(self) -> HookRegistry:
        return self._hooks

    @property
    def file_cache(self) -> FileStateCache | None:
        return self._file_cache

    @property
    def model_profile(self) -> ModelProfile:
        return self._model_profile

    @property
    def cache_stats(self) -> CacheStats:
        """Return prompt cache statistics."""
        return self._cache_manager.get_cache_stats()

    @property
    def session_memory(self) -> SessionMemoryManager | None:
        """Access the session memory manager (None if disabled)."""
        return self._session_memory

    @property
    def thinking_config(self) -> ThinkingConfig | None:
        """Current thinking/extended reasoning configuration."""
        return self._thinking

    @thinking_config.setter
    def thinking_config(self, config: ThinkingConfig | None) -> None:
        self._thinking = config

    # ── Mode switching ────────────────────────────────────────────────────────

    def enable_review_mode(
        self,
        handler: Any = None,
        auto_register_tool: bool = True,
    ) -> None:
        """
        Activate Review Mode. On first call (or when handler changes),
        registers the `request_approval` tool so the model can gate phases.

        Parameters
        ----------
        handler : ApprovalHandler | None
            Custom approval handler. If None, uses ConsoleApprovalHandler.
        auto_register_tool : bool
            If False, does not register the approval tool (user is responsible).
        """
        self._prompt_builder.enable_review_mode()
        self._review_state.reset()  # fresh state for new review session

        if not auto_register_tool:
            return

        from .tools.approval import RequestApprovalTool, ConsoleApprovalHandler
        effective_handler = handler or ConsoleApprovalHandler()

        if self._approval_tool_registered:
            # Update handler on existing tool instance
            existing = self._tool_registry.get("request_approval")
            if existing is not None:
                existing._handler = effective_handler
        else:
            tool = RequestApprovalTool(effective_handler)
            self._tool_registry.register(tool)
            self._approval_tool_registered = True

    def disable_review_mode(self) -> None:
        self._prompt_builder.disable_review_mode()
        self._review_state.reset()
        # Unregister the approval tool — prevents accidental calls in general mode
        if self._approval_tool_registered:
            self._tool_registry.unregister("request_approval")
            self._approval_tool_registered = False

    @property
    def review_mode(self) -> bool:
        return self._prompt_builder.review_mode

    @property
    def review_state(self) -> ReviewState:
        return self._review_state

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def token_usage(self) -> TokenUsage:
        return self._session.token_usage

    @property
    def messages(self) -> list:
        return list(self._session.messages)

    @property
    def session(self) -> Session:
        return self._session

    @property
    def transcript(self) -> TranscriptWriter | None:
        return self._transcript

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

    # ── Internal: API call with retry ─────────────────────────────────────────

    async def _call_with_retry(self, system: str, tools_schema: Any) -> Any:
        """Call the provider with retry logic."""
        from ._types import ProviderResponse

        # Apply prompt caching (mutates copies, not originals)
        api_messages = self._session.as_api_messages(self._provider_type)
        cached_messages, cached_system, cached_tools = (
            self._cache_manager.prepare_messages(
                api_messages, system, tools_schema, self._model_profile,
            )
        )

        # Apply thinking config if enabled
        thinking_config = self._thinking
        if thinking_config is not None:
            # Auto-adjust budget based on context usage
            thinking_config = ThinkingManager.adjust_budget(
                self._session.token_usage, thinking_config,
            )

        async def _do_call(**kwargs) -> ProviderResponse:
            max_tokens = kwargs.get("max_tokens", self._max_output_tokens)
            model = kwargs.get("model", None)

            # If fallback model specified, create a temporary provider
            provider = self._provider
            if model and model != self._model:
                # Use fallback — log it
                logger.info("Using fallback model: %s", model)

            # Build base payload for thinking injection
            call_kwargs: dict[str, Any] = {
                "messages": cached_messages,
                "system": cached_system,
                "max_tokens": max_tokens,
                "temperature": self._temperature,
                "tools": cached_tools,
            }

            # Inject thinking parameters if configured
            if thinking_config is not None:
                call_kwargs = ThinkingManager.prepare_request(
                    thinking_config, self._model_profile, call_kwargs,
                )

            return await provider.chat(**call_kwargs)

        # Wrap in persistent retry if enabled
        if self._persistent_retry_manager and self._persistent_retry:
            async def _retriable_call(**kwargs):
                r = await with_retry(
                    _do_call,
                    config=self._retry_config,
                    max_tokens=self._max_output_tokens,
                )
                return r.result

            pr_result = await self._persistent_retry_manager.execute_with_persistent_retry(
                _retriable_call,
                config=self._persistent_retry,
                initial_model=self._model,
            )
            return pr_result.result

        try:
            result = await with_retry(
                _do_call,
                config=self._retry_config,
                max_tokens=self._max_output_tokens,
            )
            return result.result
        except Exception:
            # If retry exhausted, try one direct call as last resort
            # (this handles the case where classify_error can't parse the error)
            raise

    # ── Internal: Tool execution with all features ────────────────────────────

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        on_tool_call: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> None:
        """
        Execute tool calls with:
          1. Permission checks
          2. Pre-tool hooks
          3. Concurrent execution (read parallel, write serial)
          4. Tool result truncation
          5. Post-tool hooks
          6. File cache updates
        """
        # Build execution requests, checking permissions for each
        requests: list[ToolExecRequest] = []
        denied_calls: list[ToolCall] = []

        for tc in tool_calls:
            # 1. Permission check
            perm = await self._permissions.check_and_resolve(
                tc.name, tc.arguments,
            )

            if perm.behavior == PermissionBehavior.DENY:
                logger.info("Permission denied for %s: %s", tc.name, perm.reason)
                denied_calls.append(tc)
                await _fire(on_tool_call, tc.name, tc.arguments)
                self._session.add_tool_result(
                    tool_use_id=tc.id,
                    tool_name=tc.name,
                    content=f"Permission denied: {perm.reason}",
                    is_error=True,
                )
                await _fire(on_tool_result, tc.name, f"Permission denied: {perm.reason}", True)
                continue

            if perm.behavior == PermissionBehavior.ASK:
                # ASK but no ask_fn resolved it — treat as denied
                logger.info("Permission requires confirmation for %s (no resolver)", tc.name)
                denied_calls.append(tc)
                self._session.add_tool_result(
                    tool_use_id=tc.id,
                    tool_name=tc.name,
                    content=f"Requires confirmation: {perm.reason}",
                    is_error=True,
                )
                continue

            # 2. Pre-tool hooks
            hook_result = await self._hooks.run_pre_tool_hooks(tc.name, tc.arguments)
            if hook_result.blocking_error:
                self._session.add_tool_result(
                    tool_use_id=tc.id,
                    tool_name=tc.name,
                    content=f"Blocked by hook: {hook_result.blocking_error}",
                    is_error=True,
                )
                continue

            if hook_result.behavior == HookBehavior.DENY:
                self._session.add_tool_result(
                    tool_use_id=tc.id,
                    tool_name=tc.name,
                    content=f"Denied by hook: {hook_result.message or 'no reason'}",
                    is_error=True,
                )
                continue

            # Apply updated input from hook
            arguments = hook_result.updated_input or tc.arguments

            await _fire(on_tool_call, tc.name, arguments)

            requests.append(ToolExecRequest(
                tool_call_id=tc.id,
                tool_name=tc.name,
                arguments=arguments,
            ))

        if not requests:
            return

        # 3. Execute with concurrent batching
        results = await self._tool_executor.execute_batch(requests)

        # 4. Process results
        for result in results:
            # Truncate large results
            output = truncate_tool_result(result.output, result.tool_name)

            # 5. Post-tool hooks
            if result.is_error:
                failure_hook = await self._hooks.run_post_tool_failure_hooks(
                    result.tool_name,
                    self._get_tool_arguments(result.tool_call_id, requests),
                    output,
                )
                if failure_hook.remediation:
                    output = f"{output}\n\nRemediation: {failure_hook.remediation}"
            else:
                post_hook = await self._hooks.run_post_tool_hooks(
                    result.tool_name,
                    self._get_tool_arguments(result.tool_call_id, requests),
                    output,
                    result.is_error,
                )
                if post_hook.additional_context:
                    output = output + "\n\n" + "\n".join(post_hook.additional_context)

            # 6. Update file cache on read/write
            if self._file_cache:
                self._update_file_cache(result, requests)

            # Add to session
            tool_msg = self._session.add_tool_result(
                tool_use_id=result.tool_call_id,
                tool_name=result.tool_name,
                content=output,
                is_error=result.is_error,
            )

            # Auto-save: append tool result to transcript
            if self._transcript:
                self._transcript.append_message(tool_msg)

            # Review Mode: update ReviewState based on approval tool result
            if result.tool_name == "request_approval":
                args = self._get_tool_arguments(result.tool_call_id, requests)
                self._handle_approval_result(args, output)

            await _fire(on_tool_result, result.tool_name, output, result.is_error)

    async def _execute_single_tool(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> Any:
        """Execute a single tool (called by ToolExecutor)."""
        tool = self._tool_registry.get(tool_name)
        if not tool:
            from .tools.base import ToolResult
            return ToolResult(success=False, output="", error=f"Tool '{tool_name}' not found")

        input_model = tool.input_schema(**arguments)
        return await tool.execute(input_model)

    def _handle_approval_result(self, args: dict[str, Any], output: str) -> None:
        """Parse request_approval tool output and advance ReviewState."""
        if not self._review_state.is_active:
            return

        approved = output.startswith("APPROVED")
        skip_remaining = "SKIP" in output.upper() and approved
        phase_arg = args.get("phase", "")

        if not approved:
            self._review_state.reject_current()
            return

        # phase=impl → record this function; never auto-advance to COMPLETE
        if phase_arg == "impl" or self._review_state.phase == ReviewPhase.IMPL:
            fn_name = args.get("title", "") or "<unnamed>"
            self._review_state.record_function(fn_name)
            if skip_remaining:
                self._review_state.skip_impl = True
            return

        # design / pipeline → advance phase
        self._review_state.approve_current(skip_remaining=skip_remaining)

    def _get_tool_arguments(
        self, tool_call_id: str, requests: list[ToolExecRequest],
    ) -> dict[str, Any]:
        """Find arguments for a tool call ID."""
        for req in requests:
            if req.tool_call_id == tool_call_id:
                return req.arguments
        return {}

    def _update_file_cache(
        self,
        result: ToolExecResult,
        requests: list[ToolExecRequest],
    ) -> None:
        """Update file cache based on tool results."""
        if not self._file_cache:
            return

        args = self._get_tool_arguments(result.tool_call_id, requests)

        if result.tool_name == "read_file" and not result.is_error:
            path = args.get("file_path", "")
            if path:
                self._file_cache.set(
                    path, result.output,
                    offset=args.get("offset"),
                    limit=args.get("limit"),
                )

        elif result.tool_name in ("write_file", "edit_file"):
            # Invalidate cache on write
            path = args.get("file_path", "")
            if path:
                self._file_cache.invalidate(path)

    # ── Internal: Compaction ──────────────────────────────────────────────────

    async def _maybe_compact(self, system: str) -> CompactionResult | None:
        if self._session.consecutive_compact_failures >= MAX_CONSECUTIVE_FAILURES:
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
                "Auto-compact: %d→%d tokens, %d→%d messages.",
                result.tokens_before, result.tokens_after,
                result.messages_before, result.messages_after,
            )
            return result
        except Exception as e:
            self._session.consecutive_compact_failures += 1
            logger.error("Auto-compact failed (%d/%d): %s",
                self._session.consecutive_compact_failures, MAX_CONSECUTIVE_FAILURES, e)
            return None

    async def compact(self, partial: bool = True) -> CompactionResult:
        system = await self._prompt_builder.build()
        return await self._compactor.compact(
            session=self._session, system_prompt=system, partial=partial,
        )
