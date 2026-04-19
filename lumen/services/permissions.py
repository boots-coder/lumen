"""
Permission system — rule-based safety guardrails for tool execution.

Mirrors src/utils/permissions/:
  - Read operations: silently allowed
  - Write operations: require user confirmation
  - Dangerous commands: denied outright
  - Rule engine with always-allow / always-ask / always-deny
  - Path-based and command-based matching
  - Denial tracking with circuit breaker
"""

from __future__ import annotations

import fnmatch
import logging
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionMode(str, Enum):
    """
    Session-level permission posture — mirrors Claude Code's permission modes.

    · DEFAULT       ask for write/edit/risky-bash (current lumen behavior)
    · ACCEPT_EDITS  auto-allow write_file/edit_file; bash still classified
    · PLAN          read-only research — deny all write/edit/bash
    · BYPASS        auto-allow everything (explicit opt-in, dangerous)
    """
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS = "bypass"


@dataclass
class PermissionResult:
    """Result of a permission check."""
    behavior: PermissionBehavior
    reason: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionRule:
    """A single permission rule."""
    pattern: str           # Glob pattern for path or command
    tool: str | None       # Tool name filter (None = all tools)
    behavior: PermissionBehavior
    reason: str = ""


# ── Constants ────────────────────────────────────────────────────────────────

# Tools that are always safe (read-only)
ALWAYS_ALLOW_TOOLS = frozenset({
    "read_file", "glob", "grep", "tree", "definitions",
})

# Tools that require confirmation
ASK_TOOLS = frozenset({
    "write_file", "edit_file",
})

# Dangerous bash commands — always deny
DANGEROUS_COMMANDS = frozenset({
    "rm -rf /", "rm -rf ~", "rm -rf *",
    "mkfs", "dd if=",
    ":(){:|:&};:",  # Fork bomb
    "chmod -R 777 /",
    "mv / ",
    "> /dev/sda",
})

# Dangerous command prefixes
DANGEROUS_PREFIXES = (
    "rm -rf /",
    "rm -rf ~",
    "mkfs.",
    "dd if=/dev",
    "chmod -R 777 /",
    ":(){ :",
)

# High-risk bash subcommands that need confirmation
RISKY_BASH_PATTERNS = (
    r"\brm\b.*-[rR]",          # rm with recursive flag
    r"\bgit\s+push\b",         # git push
    r"\bgit\s+reset\s+--hard", # git reset --hard
    r"\bgit\s+checkout\s+\.",  # git checkout .
    r"\bgit\s+clean\b",        # git clean
    r"\bsudo\b",               # sudo anything
    r"\bcurl\b.*\|\s*sh",      # curl | sh (pipe to shell)
    r"\bwget\b.*\|\s*sh",      # wget | sh
    r"\bnpm\s+publish\b",      # npm publish
    r"\bpip\s+install\b",      # pip install (might want confirmation)
    r"\bdocker\s+rm\b",        # docker rm
    r"\bkubectl\s+delete\b",   # kubectl delete
)

# Safe bash commands that don't need confirmation
SAFE_BASH_PATTERNS = (
    r"^(cat|head|tail|wc|sort|uniq|diff)\b",
    r"^(ls|pwd|echo|date|whoami|hostname|uname)\b",
    r"^(git\s+(status|log|diff|branch|show|rev-parse))\b",
    r"^(python|python3|node|npm\s+test|pytest|cargo\s+test|go\s+test|make\s+test)\b",
    r"^(grep|rg|find|fd|ag)\b",
    r"^(which|type|file|stat)\b",
)

# Denial tracking limits
MAX_CONSECUTIVE_DENIALS = 3
MAX_TOTAL_DENIALS = 20


# ── Denial tracker ───────────────────────────────────────────────────────────

@dataclass
class DenialTracker:
    """Tracks permission denials for circuit-breaking."""
    consecutive: int = 0
    total: int = 0
    history: list[PermissionResult] = field(default_factory=list)

    def record_denial(self, result: PermissionResult) -> None:
        self.consecutive += 1
        self.total += 1
        self.history.append(result)

    def record_allow(self) -> None:
        self.consecutive = 0

    @property
    def should_force_ask(self) -> bool:
        """Too many denials — force interactive mode."""
        return (
            self.consecutive >= MAX_CONSECUTIVE_DENIALS
            or self.total >= MAX_TOTAL_DENIALS
        )


# ── Permission checker ───────────────────────────────────────────────────────

class PermissionChecker:
    """
    Rule-based permission system with optional ML command classification.

    Check order:
      1. Custom rules (user-defined, highest priority)
      2. Built-in tool classification (read=allow, write=ask)
      3. Bash command analysis (safe=allow, risky=ask, dangerous=deny)
         When use_ml_classifier=True, the ML classifier runs alongside
         regex rules and the more restrictive result wins.

    The ask_fn callback is called when confirmation is needed.
    If None, all ASK results are returned without resolution.
    """

    def __init__(
        self,
        rules: list[PermissionRule] | None = None,
        ask_fn: Callable[[str, dict[str, Any]], Any] | None = None,
        auto_allow_reads: bool = True,
        use_ml_classifier: bool = True,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> None:
        self._rules = rules or []
        self._ask_fn = ask_fn
        self._auto_allow_reads = auto_allow_reads
        self._denial_tracker = DenialTracker()
        self._use_ml_classifier = use_ml_classifier
        self._ml_classifier: CommandClassifier | None = None
        self._mode = mode

        if use_ml_classifier:
            from .command_classifier import CommandClassifier
            self._ml_classifier = CommandClassifier()

    @property
    def denial_tracker(self) -> DenialTracker:
        return self._denial_tracker

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    def set_mode(self, mode: PermissionMode) -> PermissionMode:
        """Switch permission posture at runtime. Returns the previous mode."""
        old = self._mode
        self._mode = mode
        return old

    def add_rule(self, rule: PermissionRule) -> None:
        """Add a permission rule (prepended = highest priority)."""
        self._rules.insert(0, rule)

    def check(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PermissionResult:
        """
        Check permission for a tool call.

        Returns PermissionResult with behavior = ALLOW, ASK, or DENY.
        """
        # 1. Custom rules always win (user-defined beats mode)
        rule_result = self._check_rules(tool_name, tool_input)
        if rule_result is not None:
            return rule_result

        # 2. Read-only tools: always allow (applies even in PLAN mode)
        if self._auto_allow_reads and tool_name in ALWAYS_ALLOW_TOOLS:
            return PermissionResult(
                behavior=PermissionBehavior.ALLOW,
                reason="Read-only tool",
                tool_name=tool_name,
                tool_input=tool_input,
            )

        # 3. Mode-based short-circuits for anything that isn't a silent read
        if self._mode == PermissionMode.BYPASS:
            return PermissionResult(
                behavior=PermissionBehavior.ALLOW,
                reason="Bypass mode — all tools auto-approved",
                tool_name=tool_name, tool_input=tool_input,
            )

        if self._mode == PermissionMode.PLAN:
            # Plan mode = research only. Writes, edits, and bash are blocked.
            if tool_name in ASK_TOOLS or tool_name == "bash":
                return PermissionResult(
                    behavior=PermissionBehavior.DENY,
                    reason=(
                        "Plan mode is read-only — propose the change in your "
                        "plan and exit plan mode (/perm default or "
                        "/perm acceptEdits) before executing."
                    ),
                    tool_name=tool_name, tool_input=tool_input,
                )

        if self._mode == PermissionMode.ACCEPT_EDITS and tool_name in ASK_TOOLS:
            return PermissionResult(
                behavior=PermissionBehavior.ALLOW,
                reason="Accept-edits mode — file writes auto-approved",
                tool_name=tool_name, tool_input=tool_input,
            )

        # 4. Default tool classification
        if tool_name in ASK_TOOLS:
            return PermissionResult(
                behavior=PermissionBehavior.ASK,
                reason=f"Write operation: {tool_name}",
                tool_name=tool_name,
                tool_input=tool_input,
            )

        # 5. Bash: analyze command (mode may still coarsen — handled above)
        if tool_name == "bash":
            return self._check_bash(tool_input)

        # 6. Unknown tool: ask
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            reason=f"Unknown tool: {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
        )

    async def check_and_resolve(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PermissionResult:
        """
        Check permission and resolve ASK via ask_fn if available.

        Returns the final resolved permission.
        """
        result = self.check(tool_name, tool_input)

        if result.behavior == PermissionBehavior.ALLOW:
            self._denial_tracker.record_allow()
            return result

        if result.behavior == PermissionBehavior.DENY:
            self._denial_tracker.record_denial(result)
            return result

        # ASK — try to resolve interactively
        if result.behavior == PermissionBehavior.ASK and self._ask_fn:
            import inspect
            try:
                answer = self._ask_fn(tool_name, tool_input)
                if inspect.isawaitable(answer):
                    answer = await answer

                if answer:
                    self._denial_tracker.record_allow()
                    return PermissionResult(
                        behavior=PermissionBehavior.ALLOW,
                        reason="User approved",
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                else:
                    denied = PermissionResult(
                        behavior=PermissionBehavior.DENY,
                        reason="User denied",
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                    self._denial_tracker.record_denial(denied)
                    return denied
            except Exception as e:
                logger.error("ask_fn failed: %s", e)
                # Fall through to ASK

        return result

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check_rules(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> PermissionResult | None:
        """Check custom rules. Returns None if no rule matches."""
        for rule in self._rules:
            if rule.tool and rule.tool != tool_name:
                continue

            # Match pattern against relevant input
            target = self._get_match_target(tool_name, tool_input)
            if target and fnmatch.fnmatch(target, rule.pattern):
                return PermissionResult(
                    behavior=rule.behavior,
                    reason=rule.reason or f"Rule: {rule.pattern}",
                    tool_name=tool_name,
                    tool_input=tool_input,
                )

        return None

    def _get_match_target(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Extract the string to match rules against."""
        if tool_name == "bash":
            return tool_input.get("command", "")
        if tool_name in ("write_file", "edit_file", "read_file"):
            return tool_input.get("file_path", "")
        if tool_name == "glob":
            return tool_input.get("pattern", "")
        return str(tool_input)

    def _check_bash(self, tool_input: dict[str, Any]) -> PermissionResult:
        """Analyze a bash command for safety."""
        command = tool_input.get("command", "").strip()

        if not command:
            return PermissionResult(
                behavior=PermissionBehavior.DENY,
                reason="Empty command",
                tool_name="bash",
                tool_input=tool_input,
            )

        # Regex-based classification
        regex_result = self._check_bash_regex(command, tool_input)

        # ML classifier (if enabled): run alongside and take more restrictive
        if self._ml_classifier is not None:
            from .command_classifier import RiskLevel

            analysis = self._ml_classifier.classify(command)

            # Map ML risk level to permission behavior
            ml_behavior = self._risk_level_to_behavior(analysis.risk_level)

            ml_result = PermissionResult(
                behavior=ml_behavior,
                reason=f"ML classifier: {analysis.explanation}",
                tool_name="bash",
                tool_input=tool_input,
            )

            # Take the more restrictive of the two results
            return self._more_restrictive(regex_result, ml_result)

        return regex_result

    def _check_bash_regex(
        self, command: str, tool_input: dict[str, Any],
    ) -> PermissionResult:
        """Regex-based bash command safety analysis."""
        # Check dangerous commands (deny)
        cmd_lower = command.lower()
        for dangerous in DANGEROUS_COMMANDS:
            if dangerous in cmd_lower:
                return PermissionResult(
                    behavior=PermissionBehavior.DENY,
                    reason=f"Dangerous command: {dangerous}",
                    tool_name="bash",
                    tool_input=tool_input,
                )

        for prefix in DANGEROUS_PREFIXES:
            if cmd_lower.startswith(prefix):
                return PermissionResult(
                    behavior=PermissionBehavior.DENY,
                    reason=f"Dangerous command prefix: {prefix}",
                    tool_name="bash",
                    tool_input=tool_input,
                )

        # Check safe commands (allow)
        for pattern in SAFE_BASH_PATTERNS:
            if re.match(pattern, command.strip()):
                return PermissionResult(
                    behavior=PermissionBehavior.ALLOW,
                    reason="Safe command",
                    tool_name="bash",
                    tool_input=tool_input,
                )

        # Check risky patterns (ask)
        for pattern in RISKY_BASH_PATTERNS:
            if re.search(pattern, command):
                return PermissionResult(
                    behavior=PermissionBehavior.ASK,
                    reason=f"Risky command pattern: {pattern}",
                    tool_name="bash",
                    tool_input=tool_input,
                )

        # Default for bash: ask (bash can do anything)
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            reason="Bash command requires confirmation",
            tool_name="bash",
            tool_input=tool_input,
        )

    @staticmethod
    def _risk_level_to_behavior(risk_level: Any) -> PermissionBehavior:
        """Map an ML RiskLevel to a PermissionBehavior."""
        from .command_classifier import RiskLevel

        if risk_level == RiskLevel.SAFE:
            return PermissionBehavior.ALLOW
        if risk_level == RiskLevel.LOW:
            return PermissionBehavior.ALLOW
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return PermissionBehavior.DENY
        # MEDIUM → ask
        return PermissionBehavior.ASK

    @staticmethod
    def _more_restrictive(
        a: PermissionResult, b: PermissionResult,
    ) -> PermissionResult:
        """Return the more restrictive of two permission results."""
        order = {
            PermissionBehavior.ALLOW: 0,
            PermissionBehavior.ASK: 1,
            PermissionBehavior.DENY: 2,
        }
        if order[b.behavior] > order[a.behavior]:
            return b
        return a
