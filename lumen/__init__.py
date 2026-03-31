"""
Lumen — model-agnostic code reading & context management SDK.

Give any LLM deep code understanding with perfect memory.

    from lumen import Agent

    agent = Agent(api_key="sk-...", model="gpt-4o")
    response = await agent.chat("Explain the architecture of this codebase")
    print(response.content)
"""

from .agent import Agent
from ._types import ChatResponse, CompactionResult, Message, Role, TokenUsage
from .context.session import Session

__all__ = [
    "Agent",
    "Session",
    "ChatResponse",
    "CompactionResult",
    "Message",
    "Role",
    "TokenUsage",
]

__version__ = "0.1.0"
