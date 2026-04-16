"""
Lumen — model-agnostic coding agent SDK.

Give any LLM deep code reading AND writing capabilities
with production-grade retry, permissions, hooks, and concurrency.

    from lumen import Agent

    agent = Agent(api_key="sk-...", model="gpt-4o")
    response = await agent.chat("Fix the bug in auth.py")
    print(response.content)
"""

from .agent import Agent
from ._types import ChatResponse, CompactionResult, Message, Role, TokenUsage, ToolCall
from .context.session import Session

# Services
from .services.permissions import PermissionChecker, PermissionRule, PermissionBehavior
from .services.hooks import HookRegistry, PreToolHookResult, PostToolHookResult
from .services.retry import RetryConfig
from .services.abort import AbortController, AbortError
from .services.errors import ErrorType, ClassifiedError

# Utils
from .utils.file_state_cache import FileStateCache

__all__ = [
    "Agent",
    "Session",
    "ChatResponse",
    "CompactionResult",
    "Message",
    "Role",
    "TokenUsage",
    "ToolCall",
    # Services
    "PermissionChecker",
    "PermissionRule",
    "PermissionBehavior",
    "HookRegistry",
    "PreToolHookResult",
    "PostToolHookResult",
    "RetryConfig",
    "AbortController",
    "AbortError",
    "ErrorType",
    "ClassifiedError",
    # Utils
    "FileStateCache",
]

__version__ = "0.3.0"
