"""
ML Command Classifier — intelligent bash command safety analysis.

Instead of pure regex matching, uses a lightweight scoring model:
  1. Tokenize the command into components (executable, flags, arguments, pipes)
  2. Score each component against known-safe and known-dangerous patterns
  3. Analyze command composition (pipes, redirects, subshells)
  4. Return a risk assessment with explanation

The "ML" here is a feature-based scoring model, not a neural network —
no heavy dependencies, deterministic, and explainable.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentType(str, Enum):
    EXECUTABLE = "executable"
    FLAG = "flag"
    ARGUMENT = "argument"
    PIPE = "pipe"
    REDIRECT = "redirect"
    SUBSHELL = "subshell"


@dataclass
class CommandComponent:
    """A single parsed component of a bash command."""
    type: ComponentType
    value: str
    risk_contribution: float = 0.0


@dataclass
class CommandAnalysis:
    """Result of ML-based command analysis."""
    command: str
    risk_level: RiskLevel
    score: float  # 0.0 = perfectly safe, 1.0 = extremely dangerous
    explanation: str
    components: list[CommandComponent] = field(default_factory=list)
    flags_analysis: dict[str, Any] = field(default_factory=dict)


# ── Risk tables ──────────────────────────────────────────────────────────────

# Executable risk scores (0.0 = safe, 1.0 = critical)
EXECUTABLE_RISK: dict[str, float] = {
    # Safe — read-only / informational
    "ls": 0.0, "pwd": 0.0, "echo": 0.0, "date": 0.0, "whoami": 0.0,
    "hostname": 0.0, "uname": 0.0, "cat": 0.0, "head": 0.0, "tail": 0.0,
    "wc": 0.0, "sort": 0.0, "uniq": 0.0, "diff": 0.0, "which": 0.0,
    "type": 0.0, "file": 0.0, "stat": 0.0, "true": 0.0, "false": 0.0,
    "printf": 0.0, "env": 0.0, "printenv": 0.0, "basename": 0.0,
    "dirname": 0.0, "realpath": 0.0, "readlink": 0.0, "tee": 0.1,
    # Search tools
    "grep": 0.0, "rg": 0.0, "find": 0.0, "fd": 0.0, "ag": 0.0,
    "locate": 0.0, "tree": 0.0, "less": 0.0, "more": 0.0,
    # Version control — mostly safe
    "git": 0.1, "svn": 0.1, "hg": 0.1,
    # Build / test tools
    "python": 0.1, "python3": 0.1, "node": 0.1, "npm": 0.2,
    "npx": 0.2, "yarn": 0.2, "pnpm": 0.2,
    "pytest": 0.1, "cargo": 0.1, "go": 0.1, "make": 0.2,
    "javac": 0.1, "java": 0.1, "gcc": 0.1, "g++": 0.1,
    "rustc": 0.1, "ruby": 0.1, "perl": 0.1,
    # Package managers
    "pip": 0.3, "pip3": 0.3, "conda": 0.3, "brew": 0.3,
    "apt": 0.4, "apt-get": 0.4, "yum": 0.4, "dnf": 0.4, "pacman": 0.4,
    # Network
    "curl": 0.3, "wget": 0.3, "ssh": 0.4, "scp": 0.4, "rsync": 0.3,
    "ping": 0.1, "dig": 0.1, "nslookup": 0.1, "traceroute": 0.1,
    # Filesystem mutation
    "cp": 0.3, "mv": 0.4, "mkdir": 0.1, "touch": 0.1,
    "ln": 0.2, "install": 0.3,
    # Dangerous
    "rm": 0.7, "rmdir": 0.3, "shred": 0.8,
    "chmod": 0.5, "chown": 0.5, "chgrp": 0.5,
    "kill": 0.5, "killall": 0.6, "pkill": 0.6,
    # Very dangerous
    "sudo": 0.9, "su": 0.9, "doas": 0.9,
    "dd": 0.8, "mkfs": 0.95, "fdisk": 0.95, "parted": 0.95,
    "mount": 0.7, "umount": 0.6,
    "iptables": 0.8, "nft": 0.8,
    "systemctl": 0.6, "service": 0.6,
    # Container / orchestration
    "docker": 0.4, "podman": 0.4, "kubectl": 0.5,
    # Shells (used in pipe targets)
    "sh": 0.5, "bash": 0.5, "zsh": 0.5, "eval": 0.8,
    "exec": 0.7, "source": 0.4, "xargs": 0.3,
}

# Flag risk scores (context-sensitive)
FLAG_RISK: dict[str, float] = {
    # Generic dangerous flags
    "--force": 0.3, "-f": 0.2, "--no-preserve-root": 1.0,
    "-rf": 0.8, "-fr": 0.8, "-Rf": 0.8,
    "--hard": 0.4, "--recursive": 0.3, "-r": 0.2, "-R": 0.2,
    "--yes": 0.2, "-y": 0.2,
    "--no-verify": 0.3, "--skip-checks": 0.3,
    "--delete": 0.4, "--purge": 0.5,
    "--all": 0.2, "-a": 0.1,
    # Safe flags
    "-v": 0.0, "--verbose": 0.0, "--version": 0.0, "--help": 0.0,
    "-h": 0.0, "-n": 0.0, "--dry-run": 0.0, "-q": 0.0, "--quiet": 0.0,
    "-l": 0.0, "--list": 0.0, "--color": 0.0, "--no-color": 0.0,
}

# Context-sensitive flag escalation: (executable, flag) -> bonus risk
FLAG_CONTEXT_ESCALATION: dict[tuple[str, str], float] = {
    ("rm", "-r"): 0.3, ("rm", "-R"): 0.3,
    ("rm", "-f"): 0.3, ("rm", "--force"): 0.3,
    ("git", "--force"): 0.4, ("git", "-f"): 0.3,
    ("chmod", "-R"): 0.3,
    ("chown", "-R"): 0.3,
    ("docker", "--force"): 0.3,
    ("kubectl", "--force"): 0.3,
}

# Argument path patterns and their risk
ARGUMENT_PATH_RISK: list[tuple[str, float]] = [
    (r"^/$", 0.9),                      # Root
    (r"^/etc(/|$)", 0.4),               # System config
    (r"^/usr(/|$)", 0.4),               # System binaries
    (r"^/boot(/|$)", 0.6),              # Boot partition
    (r"^/sys(/|$)", 0.5),               # Sysfs
    (r"^/proc(/|$)", 0.4),              # Procfs
    (r"^/dev/", 0.5),                   # Devices
    (r"^/dev/null$", 0.0),             # /dev/null is fine
    (r"^/var/log(/|$)", 0.2),           # Logs
    (r"^\$HOME(/|$)", 0.2),             # Home dir
    (r"^~/", 0.2),                      # Home dir shorthand
    (r"^\*$", 0.3),                     # Glob everything
    (r"\.\.", 0.2),                     # Parent traversal
]


# ── Classifier ───────────────────────────────────────────────────────────────

class CommandClassifier:
    """
    Feature-based command safety classifier.

    Tokenizes bash commands and scores them against known patterns,
    producing a deterministic, explainable risk assessment.
    """

    # Aggregation weights
    WEIGHT_EXECUTABLE = 0.4
    WEIGHT_FLAGS = 0.2
    WEIGHT_ARGUMENTS = 0.1
    WEIGHT_COMPOSITION = 0.3

    def classify(self, command: str) -> CommandAnalysis:
        """
        Classify a bash command's risk level.

        Returns a CommandAnalysis with score, risk level, and explanation.
        """
        command = command.strip()
        if not command:
            return CommandAnalysis(
                command=command,
                risk_level=RiskLevel.SAFE,
                score=0.0,
                explanation="Empty command",
            )

        # Tokenize
        components = self._tokenize(command)

        # Extract parts
        executables = [c for c in components if c.type == ComponentType.EXECUTABLE]
        flags = [c for c in components if c.type == ComponentType.FLAG]
        arguments = [c for c in components if c.type == ComponentType.ARGUMENT]

        # Score each dimension
        exec_score = self._score_executables(executables)
        flag_score, flags_analysis = self._score_flags(flags, executables)
        arg_score = self._score_arguments(arguments)
        comp_score = self._score_composition(components, command)

        # Update risk contributions
        for c in executables:
            c.risk_contribution = EXECUTABLE_RISK.get(c.value, 0.3)
        for c in flags:
            c.risk_contribution = FLAG_RISK.get(c.value, 0.1)

        # Weighted aggregate
        score = (
            self.WEIGHT_EXECUTABLE * exec_score
            + self.WEIGHT_FLAGS * flag_score
            + self.WEIGHT_ARGUMENTS * arg_score
            + self.WEIGHT_COMPOSITION * comp_score
        )
        score = min(score, 1.0)

        # Determine risk level
        risk_level = self._score_to_risk_level(score)

        # Build explanation
        explanation = self._build_explanation(
            exec_score, flag_score, arg_score, comp_score,
            executables, flags, components, score,
        )

        return CommandAnalysis(
            command=command,
            risk_level=risk_level,
            score=round(score, 4),
            explanation=explanation,
            components=components,
            flags_analysis=flags_analysis,
        )

    # ── Tokenizer ────────────────────────────────────────────────────────────

    def _tokenize(self, command: str) -> list[CommandComponent]:
        """
        Parse a bash command into typed components.

        Handles pipes, redirects, subshells, and basic shlex tokenization.
        """
        components: list[CommandComponent] = []

        # Detect subshells
        if "$(" in command or "`" in command:
            components.append(CommandComponent(
                type=ComponentType.SUBSHELL,
                value="command_substitution",
            ))

        if command.startswith("(") or "( " in command:
            components.append(CommandComponent(
                type=ComponentType.SUBSHELL,
                value="subshell",
            ))

        # Split on pipes to handle multi-command pipelines
        pipe_segments = self._split_on_pipes(command)

        for i, segment in enumerate(pipe_segments):
            if i > 0:
                components.append(CommandComponent(
                    type=ComponentType.PIPE,
                    value="|",
                ))

            # Detect redirects in segment
            redirect_match = re.search(r"(>{1,2}|<)\s*(\S+)", segment)
            if redirect_match:
                components.append(CommandComponent(
                    type=ComponentType.REDIRECT,
                    value=redirect_match.group(0).strip(),
                ))

            # Tokenize the segment
            segment_clean = re.sub(r"(>{1,2}|<)\s*\S+", "", segment).strip()
            try:
                tokens = shlex.split(segment_clean)
            except ValueError:
                # Unmatched quotes etc. — fall back to simple split
                tokens = segment_clean.split()

            if not tokens:
                continue

            # First token is the executable
            exe = tokens[0]
            components.append(CommandComponent(
                type=ComponentType.EXECUTABLE,
                value=exe,
            ))

            # Remaining tokens: classify as flags or arguments
            for token in tokens[1:]:
                if token.startswith("-"):
                    components.append(CommandComponent(
                        type=ComponentType.FLAG,
                        value=token,
                    ))
                else:
                    components.append(CommandComponent(
                        type=ComponentType.ARGUMENT,
                        value=token,
                    ))

        return components

    def _split_on_pipes(self, command: str) -> list[str]:
        """Split command on pipe operators, respecting quotes."""
        segments: list[str] = []
        current: list[str] = []
        in_single = False
        in_double = False
        i = 0

        while i < len(command):
            ch = command[i]

            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "|" and not in_single and not in_double:
                # Check for || (logical OR) — treat as single segment
                if i + 1 < len(command) and command[i + 1] == "|":
                    current.append("||")
                    i += 2
                    continue
                segments.append("".join(current))
                current = []
                i += 1
                continue

            current.append(ch)
            i += 1

        if current:
            segments.append("".join(current))

        return [s.strip() for s in segments if s.strip()]

    # ── Scorers ──────────────────────────────────────────────────────────────

    def _score_executables(self, executables: list[CommandComponent]) -> float:
        """Score executables, taking the maximum risk across all in the pipeline."""
        if not executables:
            return 0.0

        scores: list[float] = []
        for comp in executables:
            exe = comp.value.split("/")[-1]  # Handle full paths
            score = EXECUTABLE_RISK.get(exe, 0.3)  # Unknown = moderate risk
            scores.append(score)

        return max(scores)

    def _score_flags(
        self,
        flags: list[CommandComponent],
        executables: list[CommandComponent],
    ) -> tuple[float, dict[str, Any]]:
        """Score flags with context-aware escalation."""
        if not flags:
            return 0.0, {}

        analysis: dict[str, Any] = {}
        max_score = 0.0
        exe_names = [e.value.split("/")[-1] for e in executables]

        for flag_comp in flags:
            flag = flag_comp.value
            base_score = FLAG_RISK.get(flag, 0.1)

            # Context escalation
            escalation = 0.0
            for exe in exe_names:
                key = (exe, flag)
                if key in FLAG_CONTEXT_ESCALATION:
                    escalation = max(escalation, FLAG_CONTEXT_ESCALATION[key])

            total = min(base_score + escalation, 1.0)
            analysis[flag] = {
                "base_score": base_score,
                "escalation": escalation,
                "total": total,
            }
            max_score = max(max_score, total)

        # Check combined flag patterns (e.g. -rf as separate -r and -f)
        flag_values = {f.value for f in flags}
        if "-r" in flag_values and "-f" in flag_values:
            combined = 0.8
            analysis["-r + -f"] = {"combined": combined}
            max_score = max(max_score, combined)

        return max_score, analysis

    def _score_arguments(self, arguments: list[CommandComponent]) -> float:
        """Score arguments based on path patterns."""
        if not arguments:
            return 0.0

        max_score = 0.0
        for arg_comp in arguments:
            arg = arg_comp.value
            for pattern, risk in ARGUMENT_PATH_RISK:
                if re.search(pattern, arg):
                    # Special case: /dev/null is safe even though /dev/ is risky
                    if arg == "/dev/null":
                        continue
                    max_score = max(max_score, risk)
                    arg_comp.risk_contribution = max(
                        arg_comp.risk_contribution, risk,
                    )

        return max_score

    def _score_composition(
        self,
        components: list[CommandComponent],
        raw_command: str,
    ) -> float:
        """Score structural risk from pipes, redirects, and subshells."""
        score = 0.0

        pipes = [c for c in components if c.type == ComponentType.PIPE]
        redirects = [c for c in components if c.type == ComponentType.REDIRECT]
        subshells = [c for c in components if c.type == ComponentType.SUBSHELL]
        executables = [c for c in components if c.type == ComponentType.EXECUTABLE]

        exe_names = [e.value.split("/")[-1] for e in executables]

        # Pipe to shell (curl | sh, wget | bash, etc.)
        if pipes and len(executables) >= 2:
            shell_names = {"sh", "bash", "zsh", "eval", "source", "exec"}
            for i, exe in enumerate(exe_names):
                if i > 0 and exe in shell_names:
                    score = max(score, 0.9)

        # Redirect to system files
        for redir in redirects:
            redir_target = re.search(r"[<>]+\s*(\S+)", redir.value)
            if redir_target:
                target = redir_target.group(1)
                for pattern, risk in ARGUMENT_PATH_RISK:
                    if re.search(pattern, target):
                        if target != "/dev/null":
                            score = max(score, risk + 0.2)

        # Subshell / command substitution
        if subshells:
            score = max(score, 0.3)

        # Chain operators with destructive commands
        if "&&" in raw_command or ";" in raw_command:
            score = max(score, 0.1)  # Mild bump for chaining

        # Multiple pipes increase risk slightly
        if len(pipes) >= 3:
            score = max(score, 0.2)

        return min(score, 1.0)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_risk_level(score: float) -> RiskLevel:
        """Map a numeric score to a RiskLevel."""
        if score <= 0.1:
            return RiskLevel.SAFE
        if score <= 0.3:
            return RiskLevel.LOW
        if score <= 0.5:
            return RiskLevel.MEDIUM
        if score <= 0.7:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    @staticmethod
    def _build_explanation(
        exec_score: float,
        flag_score: float,
        arg_score: float,
        comp_score: float,
        executables: list[CommandComponent],
        flags: list[CommandComponent],
        components: list[CommandComponent],
        total: float,
    ) -> str:
        """Build a human-readable explanation of the risk assessment."""
        parts: list[str] = []

        exe_names = [e.value for e in executables]
        if exe_names:
            parts.append(f"Executables: {', '.join(exe_names)} (risk={exec_score:.2f})")

        if flags:
            flag_vals = [f.value for f in flags]
            parts.append(f"Flags: {', '.join(flag_vals)} (risk={flag_score:.2f})")

        if arg_score > 0:
            parts.append(f"Argument risk: {arg_score:.2f}")

        if comp_score > 0:
            reasons: list[str] = []
            if any(c.type == ComponentType.PIPE for c in components):
                reasons.append("piped commands")
            if any(c.type == ComponentType.REDIRECT for c in components):
                reasons.append("redirects")
            if any(c.type == ComponentType.SUBSHELL for c in components):
                reasons.append("subshell/substitution")
            parts.append(
                f"Composition risk: {comp_score:.2f} ({', '.join(reasons)})"
                if reasons else f"Composition risk: {comp_score:.2f}"
            )

        parts.append(f"Total score: {total:.4f}")
        return "; ".join(parts)
