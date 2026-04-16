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
]
