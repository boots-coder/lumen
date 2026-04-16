"""Services — retry, error classification, permissions, hooks, abort, tool execution."""

from .errors import ErrorType, ClassifiedError, classify_error, parse_max_tokens_overflow
from .retry import RetryConfig, RetryResult, with_retry
from .tool_executor import ToolExecutor, ToolExecRequest, ToolExecResult, ToolStatus
from .permissions import (
    PermissionChecker, PermissionResult, PermissionBehavior,
    PermissionRule, DenialTracker,
)
from .hooks import (
    HookRegistry, PreToolHookResult, PostToolHookResult,
    PostToolFailureHookResult, HookBehavior,
)
from .abort import AbortController, AbortError
from .prompt_cache import PromptCacheManager, CacheStrategy, CacheStats
from .structured_output import (
    OutputFormat, StructuredOutputConfig, StructuredResult,
    prepare_structured_request, validate_output,
)
from .lsp import LSPClient, Location, Symbol, get_client, shutdown_all as lsp_shutdown_all
from .command_classifier import (
    CommandClassifier, CommandAnalysis, CommandComponent,
    RiskLevel, ComponentType,
)
from .skills import Skill, SkillRegistry, SkillExecutor, SkillResult
from .persistent_retry import (
    PersistentRetryConfig, PersistentRetryManager,
    PersistentRetryResult, RetryLogEntry,
)

__all__ = [
    # errors
    "ErrorType", "ClassifiedError", "classify_error", "parse_max_tokens_overflow",
    # retry
    "RetryConfig", "RetryResult", "with_retry",
    # tool executor
    "ToolExecutor", "ToolExecRequest", "ToolExecResult", "ToolStatus",
    # permissions
    "PermissionChecker", "PermissionResult", "PermissionBehavior",
    "PermissionRule", "DenialTracker",
    # hooks
    "HookRegistry", "PreToolHookResult", "PostToolHookResult",
    "PostToolFailureHookResult", "HookBehavior",
    # abort
    "AbortController", "AbortError",
    # prompt cache
    "PromptCacheManager", "CacheStrategy", "CacheStats",
    # structured output
    "OutputFormat", "StructuredOutputConfig", "StructuredResult",
    "prepare_structured_request", "validate_output",
    # lsp
    "LSPClient", "Location", "Symbol", "get_client", "lsp_shutdown_all",
    # command classifier
    "CommandClassifier", "CommandAnalysis", "CommandComponent",
    "RiskLevel", "ComponentType",
    # skills
    "Skill", "SkillRegistry", "SkillExecutor", "SkillResult",
    # persistent retry
    "PersistentRetryConfig", "PersistentRetryManager",
    "PersistentRetryResult", "RetryLogEntry",
]
