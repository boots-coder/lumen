"""
File edit tracker — captures a file's pre-edit content on the first
write/edit of the session, so we can render a session-wide unified diff
with `/diff`.

Install by subscribing to an `Agent`'s hook registry:

    tracker = FileEditTracker()
    tracker.install(agent.hooks)

    # … agent runs, writes files …

    for entry in tracker.entries():
        print(entry.path, entry.unified_diff())

Design:

 · Snapshot captured in **pre-tool** hook (before write/edit executes).
 · Only the *first* write/edit to each path is snapshotted — re-edits
   of the same file keep the original `before` content, so diffs always
   show the full session-level delta.
 · Deleted-before-session files show an empty `before`.
 · Post-tool hook just marks the entry `dirty=True` if the tool succeeded;
   the actual "after" content is read fresh from disk on demand so diffs
   reflect the current state (including external edits).
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hooks import HookRegistry, PreToolHookResult, PostToolHookResult

logger = logging.getLogger(__name__)


# Tools whose file_path argument we should snapshot.
TRACKED_TOOLS = frozenset({"write_file", "edit_file"})


@dataclass
class FileEditEntry:
    path: str
    before: str = ""          # File content at the moment of first write/edit
    existed_before: bool = False
    dirty: bool = False       # True after at least one successful write/edit
    write_count: int = 0

    def after(self) -> str:
        """Read current on-disk content (empty string if missing)."""
        try:
            return Path(self.path).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        except Exception:
            return ""

    def unified_diff(self, context_lines: int = 3) -> str:
        """Return a unified-diff string (before → current disk state)."""
        after = self.after()
        if self.before == after:
            return ""
        diff = difflib.unified_diff(
            self.before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{self.path}" + ("" if self.existed_before else "  (new file)"),
            tofile=f"b/{self.path}",
            n=context_lines,
        )
        return "".join(diff)

    def line_stats(self) -> tuple[int, int]:
        """Return (added, removed) line counts."""
        before_lines = self.before.splitlines()
        after_lines = self.after().splitlines()
        sm = difflib.SequenceMatcher(None, before_lines, after_lines)
        added = removed = 0
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "replace":
                removed += i2 - i1
                added += j2 - j1
            elif op == "delete":
                removed += i2 - i1
            elif op == "insert":
                added += j2 - j1
        return added, removed


class FileEditTracker:
    """Session-scoped file edit log."""

    def __init__(self) -> None:
        self._entries: dict[str, FileEditEntry] = {}

    # ── public API ───────────────────────────────────────────────────────────

    def install(self, hooks: HookRegistry) -> None:
        """Subscribe to a HookRegistry so every write/edit gets tracked."""
        hooks.on_pre_tool(self._on_pre_tool)
        hooks.on_post_tool(self._on_post_tool)

    def entries(self, dirty_only: bool = True) -> list[FileEditEntry]:
        """All tracked entries, sorted by path."""
        items = sorted(self._entries.values(), key=lambda e: e.path)
        if dirty_only:
            items = [e for e in items if e.dirty]
        return items

    def find(self, path: str) -> FileEditEntry | None:
        """Look up a single entry by absolute or cwd-relative path."""
        p = self._normalize(path)
        return self._entries.get(p)

    def reset(self) -> None:
        self._entries.clear()

    # ── hook callbacks ───────────────────────────────────────────────────────

    def _on_pre_tool(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> PreToolHookResult | None:
        if tool_name not in TRACKED_TOOLS:
            return None
        file_path = tool_input.get("file_path")
        if not file_path:
            return None
        key = self._normalize(file_path)
        if key in self._entries:
            return None  # already snapshotted
        try:
            p = Path(key)
            if p.exists():
                before = p.read_text(encoding="utf-8", errors="replace")
                existed = True
            else:
                before = ""
                existed = False
        except Exception as e:
            logger.debug("snapshot %s failed: %s", key, e)
            before, existed = "", False
        self._entries[key] = FileEditEntry(
            path=key, before=before, existed_before=existed,
        )
        return None  # don't block — just observe

    def _on_post_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        output: str,
        is_error: bool,
    ) -> PostToolHookResult | None:
        if is_error or tool_name not in TRACKED_TOOLS:
            return None
        file_path = tool_input.get("file_path")
        if not file_path:
            return None
        entry = self._entries.get(self._normalize(file_path))
        if entry is not None:
            entry.dirty = True
            entry.write_count += 1
        return None

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(path: str) -> str:
        """Resolve to an absolute path string for stable keying."""
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return path


if __name__ == "__main__":
    # Manual test
    import tempfile, os
    tracker = FileEditTracker()

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.py")
        with open(p, "w") as f:
            f.write("print('v1')\n")

        # Simulate pre-tool snapshot
        tracker._on_pre_tool("edit_file", {"file_path": p})
        # Simulate edit
        with open(p, "w") as f:
            f.write("print('v2')\nprint('added')\n")
        # Post-tool marks dirty
        tracker._on_post_tool("edit_file", {"file_path": p}, "ok", False)

        entries = tracker.entries()
        assert len(entries) == 1
        e = entries[0]
        assert e.dirty
        added, removed = e.line_stats()
        print(f"added={added} removed={removed}")
        print(e.unified_diff())
