"""Base tool definition and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)


class ToolResult(BaseModel):
    """Result of a tool execution."""

    success: bool
    output: str
    error: str | None = None


class Tool(ABC, Generic[InputT]):
    """
    Abstract base class for all tools.

    Each tool defines:
    - name: Unique identifier used in API calls
    - description: Help text for the model
    - input_schema: Pydantic model for input validation
    - execute: The actual tool logic

    Tools are read-only by default for code reading use cases.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used in API calls)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description (shown to model)."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> type[InputT]:
        """Input schema (Pydantic model)."""
        pass

    @abstractmethod
    async def execute(self, input_data: InputT) -> ToolResult:
        """Execute the tool and return result."""
        pass

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        schema = self.input_schema.model_json_schema()

        # Remove $defs if present (OpenAI doesn't like it)
        if "$defs" in schema:
            del schema["$defs"]

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool use schema."""
        schema = self.input_schema.model_json_schema()

        # Remove $defs if present
        if "$defs" in schema:
            del schema["$defs"]

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }
