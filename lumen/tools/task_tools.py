"""
Task observation tools — let the model manage background sub-agents.

Pairs with `SubAgentTool(run_in_background=True)` which spawns a child agent
and returns an `agent_id`. The model uses these tools to track, collect, or
cancel the background work.

  · `task_list`   — snapshot of all background tasks (id / status / duration)
  · `task_output` — fetch the final text of a completed task (or "still running")
  · `task_stop`   — cancel a running task
  · `task_wait`   — block until any background task finishes (with timeout)

All four are registered together by `Agent.enable_subagents()`, so the model
gains the full spawn→observe→harvest loop in one call.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from ..services.subagent import SubAgentManager


# ─────────────────────────────────────────────────────────────────────────────
# task_list
# ─────────────────────────────────────────────────────────────────────────────

class TaskListInput(BaseModel):
    pass


class TaskListTool(Tool[TaskListInput]):
    @property
    def name(self) -> str:
        return "task_list"

    @property
    def description(self) -> str:
        return (
            "List all background sub-agent tasks with their status. "
            "Use after spawning background agents to see what's still running."
        )

    @property
    def input_schema(self) -> type[TaskListInput]:
        return TaskListInput

    def __init__(self, manager: SubAgentManager) -> None:
        self._manager = manager

    async def execute(self, input_data: TaskListInput) -> ToolResult:
        tasks = self._manager.list_tasks()
        if not tasks:
            return ToolResult(success=True, output="No background tasks.")

        now = time.monotonic()
        lines = ["id        status     elapsed   description"]
        for t in tasks:
            elapsed = now - t.start_time if t.start_time else 0.0
            lines.append(
                f"{t.agent_id:<9} {t.status:<10} {elapsed:>6.1f}s   "
                f"{t.description or '(no description)'}"
            )
        return ToolResult(success=True, output="\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# task_output
# ─────────────────────────────────────────────────────────────────────────────

class TaskOutputInput(BaseModel):
    task_id: str = Field(description="The task id returned by sub_agent")


class TaskOutputTool(Tool[TaskOutputInput]):
    @property
    def name(self) -> str:
        return "task_output"

    @property
    def description(self) -> str:
        return (
            "Get the final output of a background sub-agent task. "
            "Returns '(still running)' if the task hasn't finished yet."
        )

    @property
    def input_schema(self) -> type[TaskOutputInput]:
        return TaskOutputInput

    def __init__(self, manager: SubAgentManager) -> None:
        self._manager = manager

    async def execute(self, input_data: TaskOutputInput) -> ToolResult:
        result = await self._manager.get_result(input_data.task_id)
        if result is None:
            tasks = {t.agent_id: t for t in self._manager.list_tasks()}
            t = tasks.get(input_data.task_id)
            if t is None:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"No task with id {input_data.task_id!r}",
                )
            return ToolResult(
                success=True,
                output=f"(still running — status={t.status})",
            )

        header = (
            f"Task {result.agent_id} · {result.description}\n"
            f"Duration: {result.duration_ms:.0f}ms · "
            f"Tool calls: {result.tool_calls_count} · "
            f"Success: {result.success}\n"
            f"{'─' * 60}\n"
        )
        body = result.content if result.success else (result.error or "")
        return ToolResult(
            success=result.success,
            output=header + body,
            error=result.error,
        )


# ─────────────────────────────────────────────────────────────────────────────
# task_stop
# ─────────────────────────────────────────────────────────────────────────────

class TaskStopInput(BaseModel):
    task_id: str = Field(description="The task id to cancel")


class TaskStopTool(Tool[TaskStopInput]):
    @property
    def name(self) -> str:
        return "task_stop"

    @property
    def description(self) -> str:
        return "Cancel a running background sub-agent task."

    @property
    def input_schema(self) -> type[TaskStopInput]:
        return TaskStopInput

    def __init__(self, manager: SubAgentManager) -> None:
        self._manager = manager

    async def execute(self, input_data: TaskStopInput) -> ToolResult:
        killed = await self._manager.kill(input_data.task_id)
        if not killed:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Could not stop {input_data.task_id!r} — "
                    f"either unknown id or already finished."
                ),
            )
        return ToolResult(
            success=True,
            output=f"Stopped task {input_data.task_id}.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# task_wait
# ─────────────────────────────────────────────────────────────────────────────

class TaskWaitInput(BaseModel):
    timeout_s: float = Field(
        default=30.0,
        description="Max seconds to wait for any background task to finish",
    )


class TaskWaitTool(Tool[TaskWaitInput]):
    @property
    def name(self) -> str:
        return "task_wait"

    @property
    def description(self) -> str:
        return (
            "Block until ANY background sub-agent finishes (or timeout). "
            "Returns the result of the first one to complete."
        )

    @property
    def input_schema(self) -> type[TaskWaitInput]:
        return TaskWaitInput

    def __init__(self, manager: SubAgentManager) -> None:
        self._manager = manager

    async def execute(self, input_data: TaskWaitInput) -> ToolResult:
        result = await self._manager.wait_any(timeout=input_data.timeout_s)
        if result is None:
            return ToolResult(
                success=True,
                output=f"(timed out after {input_data.timeout_s}s — no task finished)",
            )
        header = (
            f"Task {result.agent_id} finished · {result.description}\n"
            f"Duration: {result.duration_ms:.0f}ms · "
            f"Success: {result.success}\n"
            f"{'─' * 60}\n"
        )
        body = result.content if result.success else (result.error or "")
        return ToolResult(
            success=result.success,
            output=header + body,
            error=result.error,
        )
