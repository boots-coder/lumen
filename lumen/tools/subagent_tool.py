"""
SubAgentTool — allows the model to spawn sub-agents for independent tasks.

Not auto-registered. Enable via Agent.enable_subagents().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from ..services.subagent import SubAgentConfig, SubAgentManager


class SubAgentInput(BaseModel):
    """Input schema for the sub_agent tool."""

    prompt: str = Field(
        description="The task for the sub-agent to perform",
    )
    description: str = Field(
        default="",
        description="Short description (3-5 words)",
    )
    model: str | None = Field(
        default=None,
        description="Override model for sub-agent",
    )
    run_in_background: bool = Field(
        default=False,
        description="Run in background instead of blocking",
    )


class SubAgentTool(Tool[SubAgentInput]):
    """
    Spawn a sub-agent to handle a complex task independently.

    The sub-agent gets its own conversation, tools, and context.
    Use for: parallel research, independent subtasks, isolated experiments.
    """

    @property
    def name(self) -> str:
        return "sub_agent"

    @property
    def description(self) -> str:
        return (
            "Spawn a sub-agent to handle a task independently. "
            "The sub-agent gets its own context and tools."
        )

    @property
    def input_schema(self) -> type[SubAgentInput]:
        return SubAgentInput

    def __init__(self, agent_manager: SubAgentManager) -> None:
        self._manager = agent_manager

    async def execute(self, input_data: SubAgentInput) -> ToolResult:
        from ..services.subagent import SubAgentConfig

        config = SubAgentConfig(
            prompt=input_data.prompt,
            description=input_data.description,
            model=input_data.model,
            run_in_background=input_data.run_in_background,
        )

        if input_data.run_in_background:
            agent_id = await self._manager.spawn(config)
            return ToolResult(
                success=True,
                output=(
                    f"Background agent launched: {agent_id}\n"
                    f"Description: {config.description}\n"
                    "Use wait_agent to get results."
                ),
            )
        else:
            result = await self._manager.spawn(config)
            # result is SubAgentResult for sync spawn
            return ToolResult(
                success=result.success,
                output=(
                    f"Sub-agent completed ({result.duration_ms:.0f}ms, "
                    f"{result.tool_calls_count} tool calls)\n\n"
                    f"{result.content}"
                ),
                error=result.error,
            )
