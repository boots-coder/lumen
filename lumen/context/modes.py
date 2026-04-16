"""
Composable Mode system for the system prompt.

Modes are additive by default; conflicts are declared as data on the Mode
itself (not encoded imperatively at call sites). A ModeStack holds the set
of active modes, resolves conflicts on activation, and produces a single
prompt fragment ordered by priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Mode:
    name: str
    prompt: str
    conflicts: frozenset[str] = frozenset()
    priority: int = 0


class ModeStack:
    """Composable set of active modes with conflict handling."""

    def __init__(self, modes: Iterable[Mode] = ()) -> None:
        self._modes: dict[str, Mode] = {}
        self._active: set[str] = set()
        for mode in modes:
            self.register(mode)

    def register(self, mode: Mode) -> None:
        self._modes[mode.name] = mode

    def activate(self, name: str) -> list[str]:
        """Activate; return names of modes auto-deactivated due to conflicts.
        Raise KeyError if unknown."""
        if name not in self._modes:
            raise KeyError(name)

        mode = self._modes[name]
        deactivated: list[str] = []

        for conflict_name in mode.conflicts:
            if conflict_name in self._active:
                self._active.discard(conflict_name)
                deactivated.append(conflict_name)

        for other_name in list(self._active):
            other = self._modes.get(other_name)
            if other is not None and name in other.conflicts:
                self._active.discard(other_name)
                if other_name not in deactivated:
                    deactivated.append(other_name)

        self._active.add(name)
        return deactivated

    def deactivate(self, name: str) -> bool:
        """Deactivate; return True if was active."""
        if name in self._active:
            self._active.discard(name)
            return True
        return False

    def is_active(self, name: str) -> bool:
        return name in self._active

    def active_names(self) -> list[str]:
        """Sorted by priority, then name."""
        return sorted(
            self._active,
            key=lambda n: (self._modes[n].priority, n),
        )

    def build_prompt(self) -> str:
        """Join active mode prompts (priority order) with '\\n\\n'.
        Empty string if none."""
        names = self.active_names()
        if not names:
            return ""
        return "\n\n".join(self._modes[n].prompt for n in names)

    def __contains__(self, name: str) -> bool:
        return name in self._active

    def __repr__(self) -> str:
        active = ", ".join(self.active_names())
        registered = ", ".join(sorted(self._modes))
        return f"ModeStack(active=[{active}], registered=[{registered}])"


def build_default_stack() -> ModeStack:
    return ModeStack(DEFAULT_MODES)


_REVIEW_PROMPT = """\
# Review Mode  [ACTIVE]  ⚑ 审阅模式

You are now in **review mode**. When writing code, you MUST follow this 3-phase workflow. \
Each phase requires explicit user confirmation before proceeding to the next.

## Phase 1 — Design (设计方案)

Before writing ANY implementation code:
1. List every function/class/method you plan to create or modify
2. Show their **signatures** (name, parameters, return type) and **one-line responsibility**
3. Do NOT write implementation bodies — only signatures and docstrings
4. Wait for user confirmation: "确认" / "ok" / "继续" / "approved"

Example output format:
```
## 设计方案

1. `def parse_config(path: str) -> Config`  — 读取并验证配置文件
2. `class DataPipeline`  — 数据处理主管线
   - `def __init__(self, source: DataSource)` — 初始化数据源连接
   - `def transform(self, raw: DataFrame) -> DataFrame` — 清洗与转换
   - `def validate(self, df: DataFrame) -> list[Error]` — 校验输出
```

## Phase 2 — Data Flow (数据流)

After design is approved:
1. Show a text-based data flow diagram using arrows (→)
2. Describe what data enters, how it transforms, and what comes out
3. Mark each transformation step with the function responsible
4. Wait for user confirmation

Example:
```
raw_input (str) → parse_config() → Config object
                                      ↓
Config → DataPipeline.__init__() → connected pipeline
                                      ↓
pipeline.transform(raw_df) → cleaned DataFrame
                                      ↓
pipeline.validate(cleaned_df) → list[Error] (empty = success)
```

## Phase 3 — Incremental Implementation (逐步实现)

After data flow is approved:
1. Implement ONE function/method at a time
2. After each function, explain:
   - What data it receives (inputs)
   - What processing it performs (logic)
   - What data it produces (outputs)
   - How data changes after this step
3. Wait for user confirmation before the next function

## Escape Hatches

- User says "全部写完" / "skip" / "write all" → complete all remaining code at once
- User says "跳过设计" / "skip design" → jump to Phase 3 directly
- For trivial tasks (one-liner fixes, typos, config changes) → compress phases, \
state what you're doing but don't require multi-phase confirmation
- For non-coding tasks (explanations, searches, questions) → this mode does NOT apply, \
respond normally"""


REVIEW_MODE: Mode = Mode(
    name="review",
    prompt=_REVIEW_PROMPT,
    conflicts=frozenset(),
    priority=20,
)


DEFAULT_MODES: list[Mode] = [REVIEW_MODE]


if __name__ == "__main__":
    # 1. Empty stack
    empty = ModeStack()
    assert empty.build_prompt() == "", "empty stack should produce empty prompt"
    assert empty.active_names() == [], "empty stack should have no active names"

    # 2. Activate review
    stack = build_default_stack()
    stack.activate("review")
    prompt = stack.build_prompt()
    assert "Review Mode" in prompt, "review prompt should be present"

    # 3. Register extra mode and verify priority ordering
    stack.register(Mode(name="hint", prompt="# Hint", conflicts=frozenset(), priority=5))
    stack.activate("hint")
    prompt = stack.build_prompt()
    assert "# Hint" in prompt and "Review Mode" in prompt
    # hint priority 5 < review priority 20 → hint comes first
    assert prompt.index("# Hint") < prompt.index("Review Mode")
    assert stack.active_names() == ["hint", "review"]

    # 4. Conflict registration
    stack2 = build_default_stack()
    stack2.activate("review")
    focus = Mode(
        name="focus",
        prompt="# Focus Mode",
        conflicts=frozenset({"review"}),
        priority=15,
    )
    stack2.register(focus)
    deactivated = stack2.activate("focus")
    assert "review" in deactivated, f"review should be auto-deactivated, got {deactivated}"
    assert not stack2.is_active("review")
    assert stack2.is_active("focus")

    # 5. Unknown name raises KeyError
    try:
        stack.activate("nonexistent")
    except KeyError:
        pass
    else:
        raise AssertionError("activating unknown mode should raise KeyError")

    # 6. Deactivate inactive returns False
    fresh = build_default_stack()
    assert fresh.deactivate("review") is False
    fresh.activate("review")
    assert fresh.deactivate("review") is True

    # __contains__
    fresh.activate("review")
    assert "review" in fresh

    print("\u2713 all mode tests passed")
