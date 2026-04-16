"""
Subagent System — spawn, manage, and communicate with child agents.

Design mirrors Claude Code's AgentTool:
  - Parent agent spawns child with isolated context
  - Child gets cloned file cache, fresh abort controller (linked to parent)
  - Sync agents block parent until complete
  - Async agents run in background, notify parent on completion
  - Results returned as tool results in parent's conversation

Key isolation:
  - File cache: CLONED (no cross-contamination)
  - Abort: CHILD of parent (parent abort -> child abort, not vice versa)
  - Tools: inherited from parent (or restricted subset)
  - Session: FRESH (child has its own conversation history)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .abort import AbortController, AbortError
from .._types import TokenUsage

if TYPE_CHECKING:
    from ..agent import Agent
    from ..tools.base import Tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubAgentConfig:
    """Configuration for spawning a sub-agent."""

    prompt: str                          # Task for the child
    description: str = ""               # Short description (3-5 words)
    model: str | None = None            # Override model (None = inherit parent)
    tools: list[Tool] | None = None     # Override tools (None = inherit parent)
    system_prompt: str | None = None    # Override system prompt
    max_tool_calls: int = 20            # Tool call limit for child
    run_in_background: bool = False     # Async execution
    name: str | None = None             # Addressable name


@dataclass
class SubAgentResult:
    """Result from a completed sub-agent."""

    content: str                # Final text from child
    success: bool
    agent_id: str
    description: str
    token_usage: TokenUsage
    tool_calls_count: int
    duration_ms: float
    error: str | None = None


@dataclass
class SubAgentTask:
    """Tracks a background sub-agent."""

    agent_id: str
    description: str
    status: str  # "running", "completed", "failed", "killed"
    config: SubAgentConfig
    result: SubAgentResult | None = None
    start_time: float = 0.0
    _task: asyncio.Task | None = field(default=None, repr=False)
    _abort: AbortController | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SubAgentManager:
    """
    Manages spawning, tracking, and communicating with child agents.

    Sync spawn blocks the parent until the child completes.
    Async spawn launches a background task and returns immediately.
    """

    def __init__(self, parent_agent: Agent) -> None:
        self._parent = parent_agent
        self._tasks: dict[str, SubAgentTask] = {}  # agent_id -> task
        self._completed_queue: asyncio.Queue[SubAgentResult] = asyncio.Queue()

    # -- Public API --------------------------------------------------------

    async def spawn(self, config: SubAgentConfig) -> SubAgentResult | str:
        """
        Spawn a child agent.

        - Sync (default): blocks until child completes, returns SubAgentResult
        - Async (run_in_background=True): returns agent_id immediately
        """
        if config.run_in_background:
            return await self._run_async(config)
        return await self._run_sync(config)

    async def get_result(self, agent_id: str) -> SubAgentResult | None:
        """Get result of a background agent (None if still running)."""
        task = self._tasks.get(agent_id)
        if task is None:
            return None
        return task.result

    async def kill(self, agent_id: str) -> bool:
        """Kill a background agent. Returns True if it was running."""
        task = self._tasks.get(agent_id)
        if task is None or task.status != "running":
            return False

        # Abort the child's abort controller
        if task._abort is not None:
            task._abort.abort(reason="Killed by parent")

        # Cancel the asyncio task
        if task._task is not None and not task._task.done():
            task._task.cancel()

        task.status = "killed"
        logger.info("Killed background agent %s (%s)", agent_id, task.description)
        return True

    def list_tasks(self) -> list[SubAgentTask]:
        """List all background tasks."""
        return list(self._tasks.values())

    async def wait_any(self, timeout: float | None = None) -> SubAgentResult | None:
        """Wait for any background agent to complete."""
        try:
            return await asyncio.wait_for(
                self._completed_queue.get(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None

    async def wait_all(self, timeout: float | None = None) -> list[SubAgentResult]:
        """Wait for all background agents to complete."""
        running = [
            t for t in self._tasks.values() if t.status == "running"
        ]
        if not running:
            return []

        async_tasks = [
            t._task for t in running if t._task is not None and not t._task.done()
        ]
        if not async_tasks:
            return [t.result for t in running if t.result is not None]

        try:
            await asyncio.wait_for(
                asyncio.gather(*async_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("wait_all timed out with %d tasks still running", len(async_tasks))

        return [
            t.result for t in self._tasks.values()
            if t.result is not None
        ]

    async def cleanup(self) -> None:
        """Kill all running agents and cleanup."""
        for agent_id in list(self._tasks):
            await self.kill(agent_id)

        # Drain the completed queue
        while not self._completed_queue.empty():
            try:
                self._completed_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._tasks.clear()

    # -- Internal ----------------------------------------------------------

    def _create_child_agent(
        self, config: SubAgentConfig,
    ) -> tuple[Agent, AbortController]:
        """Create an isolated child Agent instance."""
        from ..agent import Agent

        # 1. Create child abort controller linked to parent
        child_abort = self._parent._abort.create_child()

        # 2. Clone file cache from parent
        child_cache = (
            self._parent._file_cache.clone()
            if self._parent._file_cache
            else None
        )

        # 3. Determine tools (inherit or override)
        if config.tools is not None:
            tools = list(config.tools)
        else:
            tools = [
                self._parent._tool_registry.get(n)
                for n in self._parent._tool_registry.list_tools()
            ]
            tools = [t for t in tools if t is not None]

        # 4. Extract API key from the parent's provider
        api_key = self._extract_api_key()

        # 5. Create child Agent with isolated context
        child = Agent(
            api_key=api_key,
            model=config.model or self._parent._model,
            base_url=self._parent._base_url,
            system_prompt=config.system_prompt,
            tools=tools,
            max_tool_calls=config.max_tool_calls,
            inject_git_state=False,
            inject_memory=False,
            auto_compact=True,
            session_memory=False,
        )

        # 6. Override abort controller and file cache
        child._abort = child_abort
        if child_cache:
            child._file_cache = child_cache

        return child, child_abort

    def _extract_api_key(self) -> str:
        """Extract the API key from the parent's provider headers."""
        provider = self._parent._provider
        headers = getattr(provider, "_headers", {})

        # OpenAI-compat: "Authorization: Bearer <key>"
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[len("Bearer "):]

        # Anthropic: "x-api-key: <key>"
        api_key = headers.get("x-api-key", "")
        if api_key:
            return api_key

        logger.warning("Could not extract API key from parent provider")
        return ""

    async def _run_child(self, child: Agent, config: SubAgentConfig) -> SubAgentResult:
        """Run a child agent to completion and return the result."""
        agent_id = config.name or str(uuid.uuid4())[:8]
        start = time.monotonic()
        tool_calls_count = 0

        try:
            response = await child.chat(config.prompt)
            duration_ms = (time.monotonic() - start) * 1000

            # Count tool calls from the child session
            for msg in child.messages:
                if getattr(msg, "tool_calls", None):
                    tool_calls_count += len(msg.tool_calls)

            return SubAgentResult(
                content=response.content,
                success=True,
                agent_id=agent_id,
                description=config.description,
                token_usage=response.token_usage,
                tool_calls_count=tool_calls_count,
                duration_ms=duration_ms,
            )

        except AbortError as e:
            duration_ms = (time.monotonic() - start) * 1000
            return SubAgentResult(
                content="",
                success=False,
                agent_id=agent_id,
                description=config.description,
                token_usage=child.token_usage,
                tool_calls_count=tool_calls_count,
                duration_ms=duration_ms,
                error=f"Aborted: {e.reason}",
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error("Sub-agent %s failed: %s", agent_id, e, exc_info=True)
            return SubAgentResult(
                content="",
                success=False,
                agent_id=agent_id,
                description=config.description,
                token_usage=child.token_usage,
                tool_calls_count=tool_calls_count,
                duration_ms=duration_ms,
                error=str(e),
            )

    async def _run_sync(self, config: SubAgentConfig) -> SubAgentResult:
        """Run child synchronously (blocking)."""
        child, _abort = self._create_child_agent(config)
        return await self._run_child(child, config)

    async def _run_async(self, config: SubAgentConfig) -> str:
        """Run child in background, return agent_id."""
        agent_id = config.name or str(uuid.uuid4())[:8]
        child, child_abort = self._create_child_agent(config)

        task_record = SubAgentTask(
            agent_id=agent_id,
            description=config.description,
            status="running",
            config=config,
            start_time=time.monotonic(),
            _abort=child_abort,
        )

        async def _background() -> None:
            try:
                result = await self._run_child(child, config)
                result.agent_id = agent_id  # ensure consistent id
                task_record.result = result
                task_record.status = "completed" if result.success else "failed"
                await self._completed_queue.put(result)
            except asyncio.CancelledError:
                task_record.status = "killed"
                result = SubAgentResult(
                    content="",
                    success=False,
                    agent_id=agent_id,
                    description=config.description,
                    token_usage=child.token_usage,
                    tool_calls_count=0,
                    duration_ms=(time.monotonic() - task_record.start_time) * 1000,
                    error="Cancelled",
                )
                task_record.result = result
                await self._completed_queue.put(result)

        bg_task = asyncio.create_task(_background())
        task_record._task = bg_task
        self._tasks[agent_id] = task_record

        logger.info(
            "Launched background agent %s: %s", agent_id, config.description,
        )
        return agent_id
