"""
Transcript Persistence — auto-save conversation history to JSONL.

Every message (user, assistant, tool call, tool result) is automatically
appended to a .jsonl file. Sessions can be resumed by loading the file.

Storage: ~/.lumen/projects/{sanitized_cwd}/{session_id}.jsonl
Format: One JSON object per line (append-only, crash-safe)

Design mirrors Claude Code's sessionStorage.ts:
  - Path sanitization (non-alphanumeric -> hyphen)
  - UUID session IDs
  - Buffered async writes (flush every 100ms or on exit)
  - Metadata entries (title, last-prompt) re-appended at EOF for fast resume
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .._types import Message, Role, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_LUMEN_HOME = Path.home() / ".lumen"
_MAX_PATH_COMPONENT = 200


def sanitize_path(path: str) -> str:
    """
    Replace non-alphanumeric chars with ``-``, collapse runs, strip edges.

    If the result exceeds 200 chars, truncate and append a short hash suffix
    so different long paths don't collide.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9]", "-", path)
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")

    if len(sanitized) > _MAX_PATH_COMPONENT:
        suffix = hashlib.sha256(path.encode()).hexdigest()[:12]
        sanitized = sanitized[: _MAX_PATH_COMPONENT - 13] + "-" + suffix

    return sanitized


def get_transcript_dir(cwd: Path) -> Path:
    """Return ``~/.lumen/projects/{sanitize_path(cwd)}/``."""
    return _LUMEN_HOME / "projects" / sanitize_path(str(cwd))


# ---------------------------------------------------------------------------
# SessionInfo
# ---------------------------------------------------------------------------

@dataclass
class SessionInfo:
    """Lightweight summary returned by :func:`list_recent_sessions`."""
    session_id: str
    model: str
    created_at: str
    last_prompt: str
    message_count: int
    file_path: Path


# ---------------------------------------------------------------------------
# TranscriptWriter
# ---------------------------------------------------------------------------

class TranscriptWriter:
    """
    Append-only JSONL writer with buffered async flushes.

    Entries are buffered in memory and flushed to disk either when
    :pymethod:`flush` is called explicitly, after a 100 ms timer, or when
    :pymethod:`close` is invoked.
    """

    def __init__(self, session_id: str, cwd: Path, model: str) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.model = model
        self.created_at = datetime.utcnow().isoformat()

        self._dir = get_transcript_dir(cwd)
        self._path = self._dir / f"{session_id}.jsonl"
        self._buffer: list[dict[str, Any]] = []
        self._flush_handle: asyncio.TimerHandle | None = None
        self._dir_created = False
        self._last_prompt: str | None = None
        self._custom_title: str | None = None
        self._message_count = 0
        self._version = "0.4.0"

        # Write the initial metadata entry
        self.append({
            "type": "metadata",
            "session_id": self.session_id,
            "model": self.model,
            "cwd": str(self.cwd),
            "created_at": self.created_at,
            "version": self._version,
        })

    @property
    def path(self) -> Path:
        return self._path

    # -- public API --------------------------------------------------------

    def append(self, entry: dict[str, Any]) -> None:
        """Add *entry* to the write buffer and schedule a flush."""
        self._buffer.append(entry)

        # Track counters for fast resume metadata
        if entry.get("type") == "message":
            self._message_count += 1
            if entry.get("role") == "user":
                content = entry.get("content", "")
                if content:
                    self._last_prompt = content[:200]

        self._schedule_flush()

    def append_message(
        self,
        msg: Message,
        *,
        parent_uuid: str | None = None,
        model: str | None = None,
        cwd: Path | None = None,
    ) -> None:
        """Convenience: build a ``message`` entry from a :class:`Message`."""
        entry: dict[str, Any] = {
            "type": "message",
            "session_id": self.session_id,
            "uuid": str(uuid.uuid4()),
            "parent_uuid": parent_uuid,
            "role": msg.role.value,
            "content": msg.content,
            "tool_calls": (
                [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                 for tc in msg.tool_calls]
                if msg.tool_calls else None
            ),
            "tool_call_id": msg.tool_call_id,
            "tool_name": msg.tool_name,
            "timestamp": msg.created_at.isoformat(),
            "model": model or self.model,
            "cwd": str(cwd or self.cwd),
        }
        self.append(entry)

    def set_title(self, title: str) -> None:
        """Store a custom title for this session."""
        self._custom_title = title
        self.append({
            "type": "custom-title",
            "session_id": self.session_id,
            "custom_title": title,
        })

    # -- flush logic -------------------------------------------------------

    def _schedule_flush(self) -> None:
        """Schedule a flush 100 ms from now (debounced)."""
        if self._flush_handle is not None:
            return  # already scheduled

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — flush synchronously
            self._flush_sync()
            return

        self._flush_handle = loop.call_later(0.1, self._flush_callback)

    def _flush_callback(self) -> None:
        """Timer callback: kick off the async flush as a task."""
        self._flush_handle = None
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.flush())
        except RuntimeError:
            self._flush_sync()

    async def flush(self) -> None:
        """Write all buffered entries to the JSONL file."""
        if not self._buffer:
            return
        self._flush_handle = None
        entries = self._buffer[:]
        self._buffer.clear()
        await asyncio.to_thread(self._write_lines, entries)

    def _flush_sync(self) -> None:
        """Synchronous fallback when no event loop is available."""
        if not self._buffer:
            return
        entries = self._buffer[:]
        self._buffer.clear()
        self._write_lines(entries)

    def _write_lines(self, entries: list[dict[str, Any]]) -> None:
        """Low-level: append JSON lines to disk."""
        self._ensure_dir()
        lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)

        fd = os.open(
            str(self._path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(fd, lines.encode("utf-8"))
        finally:
            os.close(fd)

    def _ensure_dir(self) -> None:
        if not self._dir_created:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._dir_created = True

    # -- close -------------------------------------------------------------

    def close(self) -> None:
        """
        Flush remaining buffer and re-append metadata entries at EOF
        for fast resume scanning.
        """
        self._flush_sync()

        # Re-append fast-resume entries at the end of the file
        tail_entries: list[dict[str, Any]] = []

        if self._last_prompt:
            tail_entries.append({
                "type": "last-prompt",
                "session_id": self.session_id,
                "last_prompt": self._last_prompt,
            })

        if self._custom_title:
            tail_entries.append({
                "type": "custom-title",
                "session_id": self.session_id,
                "custom_title": self._custom_title,
            })

        # Always re-append metadata with updated message count
        tail_entries.append({
            "type": "metadata",
            "session_id": self.session_id,
            "model": self.model,
            "cwd": str(self.cwd),
            "created_at": self.created_at,
            "version": self._version,
            "message_count": self._message_count,
        })

        if tail_entries:
            self._buffer = tail_entries
            self._flush_sync()

        logger.debug(
            "Transcript closed: %s (%d messages)", self._path, self._message_count,
        )


# ---------------------------------------------------------------------------
# TranscriptReader
# ---------------------------------------------------------------------------

class TranscriptReader:
    """Read JSONL transcript files back into messages and metadata."""

    @staticmethod
    def load_session(path: Path) -> tuple[list[Message], dict[str, Any]]:
        """
        Read a JSONL transcript and reconstruct messages + metadata.

        Returns
        -------
        (messages, metadata) where *metadata* is the last ``metadata`` entry
        merged with the last ``last-prompt`` and ``custom-title`` entries.
        """
        messages: list[Message] = []
        metadata: dict[str, Any] = {}
        last_prompt: str | None = None
        custom_title: str | None = None

        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", path)
                    continue

                entry_type = entry.get("type")

                if entry_type == "message":
                    msg = _entry_to_message(entry)
                    if msg is not None:
                        messages.append(msg)

                elif entry_type == "metadata":
                    metadata = entry

                elif entry_type == "last-prompt":
                    last_prompt = entry.get("last_prompt")

                elif entry_type == "custom-title":
                    custom_title = entry.get("custom_title")

        if last_prompt:
            metadata["last_prompt"] = last_prompt
        if custom_title:
            metadata["custom_title"] = custom_title
        metadata.setdefault("message_count", len(messages))

        return messages, metadata

    @staticmethod
    def list_sessions(cwd: Path) -> list[SessionInfo]:
        """
        Scan the project transcript directory for ``.jsonl`` files.

        For efficiency, only the last 64 KB of each file is read to
        extract the trailing metadata / last-prompt entries.
        """
        transcript_dir = get_transcript_dir(cwd)
        if not transcript_dir.is_dir():
            return []

        sessions: list[SessionInfo] = []
        for fp in transcript_dir.glob("*.jsonl"):
            info = _read_session_info(fp)
            if info is not None:
                sessions.append(info)

        # Sort by modification time, newest first
        sessions.sort(key=lambda s: s.file_path.stat().st_mtime, reverse=True)
        return sessions


def list_recent_sessions(cwd: Path, limit: int = 10) -> list[SessionInfo]:
    """Return the *limit* most recent sessions for *cwd*."""
    return TranscriptReader.list_sessions(cwd)[:limit]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entry_to_message(entry: dict[str, Any]) -> Message | None:
    """Convert a ``message`` transcript entry back to a :class:`Message`."""
    role_str = entry.get("role")
    if not role_str:
        return None

    try:
        role = Role(role_str)
    except ValueError:
        return None

    tool_calls: list[ToolCall] | None = None
    tc_data = entry.get("tool_calls")
    if tc_data:
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
            )
            for tc in tc_data
        ]

    return Message(
        role=role,
        content=entry.get("content", "") or "",
        tool_calls=tool_calls,
        tool_call_id=entry.get("tool_call_id"),
        tool_name=entry.get("tool_name"),
    )


_TAIL_BYTES = 64 * 1024  # 64 KB


def _read_session_info(fp: Path) -> SessionInfo | None:
    """
    Read trailing metadata from a transcript file (last 64 KB).

    This avoids reading the entire (potentially large) file just to
    list sessions.
    """
    try:
        size = fp.stat().st_size
        if size == 0:
            return None

        with open(fp, "rb") as fh:
            if size > _TAIL_BYTES:
                fh.seek(size - _TAIL_BYTES)
                # Skip the first partial line
                fh.readline()
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    session_id = fp.stem
    model = ""
    created_at = ""
    last_prompt = ""
    message_count = 0

    for line in tail.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type")
        if entry_type == "metadata":
            model = entry.get("model", model)
            created_at = entry.get("created_at", created_at)
            message_count = entry.get("message_count", message_count)
        elif entry_type == "last-prompt":
            last_prompt = entry.get("last_prompt", last_prompt)

    return SessionInfo(
        session_id=session_id,
        model=model,
        created_at=created_at,
        last_prompt=last_prompt,
        message_count=message_count,
        file_path=fp,
    )
