"""
Review Mode state machine — tracks which phase of the 3-phase review workflow
the agent is currently in, across turns.

Review Mode walks the agent through a gated coding workflow:
  Phase 1 (DESIGN)   — propose signatures / responsibilities, await approval
  Phase 2 (PIPELINE) — draw the data-flow diagram, await approval
  Phase 3 (IMPL)     — implement one function at a time (or all at once if skipped)

The Agent holds a ReviewState instance and injects `as_system_reminder()` into
each turn so the model always knows its exact position in the workflow.

State transition matrix
-----------------------
  from\\to    IDLE   DESIGN  PIPELINE  IMPL  COMPLETE
  IDLE       —      start() —         —     —
  DESIGN     reset  —       approve   appr* —
  PIPELINE   reset  —       —         appr  —
  IMPL       reset  —       —         —     complete()
  COMPLETE   reset  start() —         —     —

  appr* = approve_current(skip_remaining=True) from DESIGN jumps to IMPL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ═════════════════════════════════════════════════════════════════════════════
# Types
# ═════════════════════════════════════════════════════════════════════════════

class ReviewPhase(str, Enum):
    """The phase of Review Mode the agent is currently in."""
    IDLE = "idle"            # review mode off, or before first task
    DESIGN = "design"        # Phase 1: awaiting design approval
    PIPELINE = "pipeline"    # Phase 2: awaiting data-flow approval
    IMPL = "impl"            # Phase 3: incremental implementation in progress
    COMPLETE = "complete"    # all phases done for current task


# ═════════════════════════════════════════════════════════════════════════════
# ReviewState
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ReviewState:
    """
    Persistent state machine for Review Mode.

    Held by the Agent and carried across turns. The `as_system_reminder()`
    output is injected into the prompt each turn so the model always knows
    which phase it is in and what it must do next.
    """

    phase: ReviewPhase = ReviewPhase.IDLE
    design_approved: bool = False
    pipeline_approved: bool = False
    functions_approved: list[str] = field(default_factory=list)
    skip_pipeline: bool = False
    skip_impl: bool = False
    task_summary: str = ""

    # ── State transitions ────────────────────────────────────────────────────

    def start(self, task_summary: str = "") -> None:
        """IDLE/COMPLETE → DESIGN. Reset approval flags."""
        self.phase = ReviewPhase.DESIGN
        self.design_approved = False
        self.pipeline_approved = False
        self.functions_approved = []
        self.skip_pipeline = False
        self.skip_impl = False
        self.task_summary = task_summary

    def approve_current(self, skip_remaining: bool = False) -> None:
        """
        Advance to the next phase.

        DESIGN   → PIPELINE  (or → IMPL if skip_remaining=True, skipping Phase 2)
        PIPELINE → IMPL
        IMPL     → (stays in IMPL; COMPLETE is reached only via complete())

        Per-function approvals in IMPL do NOT auto-advance to COMPLETE — the
        caller is responsible for deciding when the whole task is finished.
        """
        if self.phase == ReviewPhase.DESIGN:
            self.design_approved = True
            if skip_remaining:
                self.skip_pipeline = True
                self.phase = ReviewPhase.IMPL
            else:
                self.phase = ReviewPhase.PIPELINE
        elif self.phase == ReviewPhase.PIPELINE:
            self.pipeline_approved = True
            self.phase = ReviewPhase.IMPL
        # IMPL and terminal states: no phase change here — record_function()
        # handles per-function progress; complete() finishes the task.

    def reject_current(self) -> None:
        """Stay in the current phase (no-op for state; caller re-prompts)."""
        # Intentionally a no-op — the caller decides how to re-prompt the model.
        return

    def record_function(self, name: str) -> None:
        """Add an approved function name to the list. Phase stays IMPL."""
        if name and name not in self.functions_approved:
            self.functions_approved.append(name)

    def complete(self) -> None:
        """Explicitly mark the current task done: phase = COMPLETE."""
        self.phase = ReviewPhase.COMPLETE

    def reset(self) -> None:
        """Full reset to IDLE; clear every flag and list."""
        self.phase = ReviewPhase.IDLE
        self.design_approved = False
        self.pipeline_approved = False
        self.functions_approved = []
        self.skip_pipeline = False
        self.skip_impl = False
        self.task_summary = ""

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True when phase is neither IDLE nor COMPLETE."""
        return self.phase not in (ReviewPhase.IDLE, ReviewPhase.COMPLETE)

    def as_system_reminder(self) -> str | None:
        """
        Return a <system-reminder>-wrapped string describing the current
        position and expected next action, or None when phase is IDLE
        (no reminder needed when review is inactive).
        """
        if self.phase == ReviewPhase.IDLE:
            return None

        if self.phase == ReviewPhase.DESIGN:
            return _DESIGN_REMINDER.format(task_summary=self.task_summary or "(unspecified)")

        if self.phase == ReviewPhase.PIPELINE:
            if self.skip_pipeline:
                # Pipeline was skipped — no reminder for this phase.
                return None
            return _PIPELINE_REMINDER

        if self.phase == ReviewPhase.IMPL:
            count = len(self.functions_approved)
            if self.skip_impl:
                body = _IMPL_BODY_SKIP
            else:
                body = _IMPL_BODY_PER_FUNCTION
            return _IMPL_REMINDER.format(count=count, body=body)

        if self.phase == ReviewPhase.COMPLETE:
            return _COMPLETE_REMINDER.format(task_summary=self.task_summary or "(unspecified)")

        return None


# ═════════════════════════════════════════════════════════════════════════════
# Reminder templates
# ═════════════════════════════════════════════════════════════════════════════

_DESIGN_REMINDER = """<system-reminder>
You are in REVIEW MODE — Phase 1 of 3 (DESIGN).

Task: {task_summary}

Your next actions:
1. Propose the function/class signatures (names, parameters, return types, one-line responsibility)
2. Do NOT write implementation bodies yet
3. Call `request_approval(phase="design", title=..., summary=<your design>)` to gate Phase 2

Wait for the approval tool result before writing any implementation code.
</system-reminder>"""


_PIPELINE_REMINDER = """<system-reminder>
You are in REVIEW MODE — Phase 2 of 3 (DATA PIPELINE).

Design approved. Now:
1. Draw a text-based data flow diagram using arrows (→)
2. Mark each transformation step with the function responsible
3. Call `request_approval(phase="pipeline", title=..., summary=<diagram>)` to gate Phase 3

Do NOT start writing code yet.
</system-reminder>"""


_IMPL_BODY_SKIP = "User has skipped per-function approval. Implement all remaining functions at once."


_IMPL_BODY_PER_FUNCTION = (
    "Implement ONE function at a time. After each, explain:\n"
    " - Inputs the function receives\n"
    " - How it transforms the data\n"
    " - Outputs it produces\n"
    "\n"
    "Then call `request_approval(phase='impl', title=<function name>, summary=<body + explanation>)`."
)


_IMPL_REMINDER = """<system-reminder>
You are in REVIEW MODE — Phase 3 of 3 (IMPLEMENTATION).

Approved so far: {count} function(s).

{body}
</system-reminder>"""


_COMPLETE_REMINDER = """<system-reminder>
Review Mode session COMPLETE for task: {task_summary}
You may respond to follow-up questions normally. The next coding task will start Phase 1 again.
</system-reminder>"""


# ═════════════════════════════════════════════════════════════════════════════
# Self-tests
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Fresh ReviewState → phase=IDLE, is_active=False, reminder is None
    s = ReviewState()
    assert s.phase == ReviewPhase.IDLE, f"expected IDLE, got {s.phase}"
    assert s.is_active is False, "fresh state should not be active"
    assert s.as_system_reminder() is None, "IDLE should have no reminder"

    # 2. start("build parser") → phase=DESIGN, task_summary set, reminder contains "Phase 1"
    s.start("build parser")
    assert s.phase == ReviewPhase.DESIGN, f"expected DESIGN, got {s.phase}"
    assert s.task_summary == "build parser", f"task_summary not set: {s.task_summary!r}"
    assert s.is_active is True, "DESIGN should be active"
    reminder = s.as_system_reminder()
    assert reminder is not None, "DESIGN reminder should not be None"
    assert "Phase 1" in reminder, "DESIGN reminder should mention Phase 1"
    assert "build parser" in reminder, "DESIGN reminder should include task summary"

    # 3. approve_current() → phase=PIPELINE, design_approved=True
    s.approve_current()
    assert s.phase == ReviewPhase.PIPELINE, f"expected PIPELINE, got {s.phase}"
    assert s.design_approved is True, "design_approved should be True"
    pipeline_reminder = s.as_system_reminder()
    assert pipeline_reminder is not None and "Phase 2" in pipeline_reminder

    # 4. approve_current() → phase=IMPL, pipeline_approved=True
    s.approve_current()
    assert s.phase == ReviewPhase.IMPL, f"expected IMPL, got {s.phase}"
    assert s.pipeline_approved is True, "pipeline_approved should be True"
    impl_reminder = s.as_system_reminder()
    assert impl_reminder is not None and "Phase 3" in impl_reminder

    # 5. record_function("parse_config") → phase stays IMPL, in functions_approved
    s.record_function("parse_config")
    assert s.phase == ReviewPhase.IMPL, f"phase should stay IMPL, got {s.phase}"
    assert "parse_config" in s.functions_approved, "parse_config should be recorded"
    # Re-recording should not duplicate
    s.record_function("parse_config")
    assert s.functions_approved.count("parse_config") == 1, "no duplicates"

    # 6. Fresh state → start() → approve_current(skip_remaining=True) →
    #    phase=IMPL (skipped pipeline), skip_pipeline=True
    s2 = ReviewState()
    s2.start("quick fix")
    s2.approve_current(skip_remaining=True)
    assert s2.phase == ReviewPhase.IMPL, f"skip_remaining should jump to IMPL, got {s2.phase}"
    assert s2.skip_pipeline is True, "skip_pipeline should be True"
    assert s2.design_approved is True, "design_approved should still be True after skip"

    # 7. reset() → phase=IDLE, all flags cleared
    s2.record_function("foo")
    s2.reset()
    assert s2.phase == ReviewPhase.IDLE, f"expected IDLE after reset, got {s2.phase}"
    assert s2.design_approved is False
    assert s2.pipeline_approved is False
    assert s2.functions_approved == []
    assert s2.skip_pipeline is False
    assert s2.skip_impl is False
    assert s2.task_summary == ""
    assert s2.is_active is False
    assert s2.as_system_reminder() is None

    # Bonus: complete() → phase=COMPLETE, reminder mentions COMPLETE
    s.complete()
    assert s.phase == ReviewPhase.COMPLETE
    assert s.is_active is False, "COMPLETE should not be active"
    done_reminder = s.as_system_reminder()
    assert done_reminder is not None and "COMPLETE" in done_reminder

    # Bonus: COMPLETE → start() restarts cycle
    s.start("next task")
    assert s.phase == ReviewPhase.DESIGN
    assert s.task_summary == "next task"
    assert s.design_approved is False

    print("\u2713 all review state tests passed")
