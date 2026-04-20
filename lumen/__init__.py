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
from .services.permissions import PermissionChecker, PermissionRule, PermissionBehavior, PermissionMode
from .services.hooks import HookRegistry, PreToolHookResult, PostToolHookResult
from .services.retry import RetryConfig
from .services.abort import AbortController, AbortError
from .services.errors import ErrorType, ClassifiedError
from .services.prompt_cache import PromptCacheManager, CacheStrategy, CacheStats
from .services.structured_output import (
    OutputFormat, StructuredOutputConfig, StructuredResult,
)
from .services.thinking import ThinkingConfig, ThinkingMode, ThinkingResult
from .services.command_classifier import (
    CommandClassifier, CommandAnalysis, CommandComponent, RiskLevel, ComponentType,
)
from .services.skills import Skill, SkillRegistry, SkillExecutor, SkillResult
from .services.persistent_retry import (
    PersistentRetryConfig, PersistentRetryManager, PersistentRetryResult, RetryLogEntry,
)
from .services.subagent import SubAgentManager, SubAgentConfig, SubAgentResult
from .services.review_state import ReviewState, ReviewPhase
from .services.notifier import Notifier
from .services.file_edit_tracker import FileEditTracker, FileEditEntry
from .services.mcp import (
    MCPClient, MCPError, MCPTool,
    MCPManager, MCPServerConfig, MCPServerState,
)
from .services.auto_dream import AutoDreamService, AutoDreamConfig, DreamStats

# Context
from .context.session_memory import SessionMemoryManager, MemoryCategory, MemoryEntry
from .context.transcript import TranscriptWriter, TranscriptReader, SessionInfo
from .context.modes import Mode, ModeStack, REVIEW_MODE, build_default_stack

# Utils
from .utils.file_state_cache import FileStateCache

# i18n (runtime English/Chinese switching for the chat UI)
from . import i18n as i18n

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
    "PermissionMode",
    "HookRegistry",
    "PreToolHookResult",
    "PostToolHookResult",
    "RetryConfig",
    "AbortController",
    "AbortError",
    "ErrorType",
    "ClassifiedError",
    # Prompt Cache
    "PromptCacheManager",
    "CacheStrategy",
    "CacheStats",
    # Structured Output
    "OutputFormat",
    "StructuredOutputConfig",
    "StructuredResult",
    # Thinking / Extended Reasoning
    "ThinkingConfig",
    "ThinkingMode",
    "ThinkingResult",
    # Session Memory
    "SessionMemoryManager",
    "MemoryCategory",
    "MemoryEntry",
    # Command Classifier
    "CommandClassifier",
    "CommandAnalysis",
    "CommandComponent",
    "RiskLevel",
    "ComponentType",
    # Skills
    "Skill",
    "SkillRegistry",
    "SkillExecutor",
    "SkillResult",
    # Persistent Retry
    "PersistentRetryConfig",
    "PersistentRetryManager",
    "PersistentRetryResult",
    "RetryLogEntry",
    # Transcript
    "TranscriptWriter",
    "TranscriptReader",
    "SessionInfo",
    # Sub-agents
    "SubAgentManager",
    "SubAgentConfig",
    "SubAgentResult",
    # Review Mode (state machine)
    "ReviewState",
    "ReviewPhase",
    # Desktop notifications
    "Notifier",
    # File edit tracking
    "FileEditTracker",
    "FileEditEntry",
    # MCP (Model Context Protocol) client
    "MCPClient",
    "MCPError",
    "MCPTool",
    "MCPManager",
    "MCPServerConfig",
    "MCPServerState",
    # Auto-Dream (background memory consolidation)
    "AutoDreamService",
    "AutoDreamConfig",
    "DreamStats",
    # Modes (composable system-prompt layers)
    "Mode",
    "ModeStack",
    "REVIEW_MODE",
    "build_default_stack",
    # Utils
    "FileStateCache",
    # i18n
    "i18n",
]

__version__ = "0.5.0"
