"""
File state LRU cache — tracks file contents for diff detection.

Mirrors src/utils/fileStateCache.ts:
  - LRU eviction: 100 files max, 25MB total size cap
  - Path normalization (resolve .., unify separators)
  - Partial view tracking (auto-injected files vs full reads)
  - Content-based size tracking for eviction
  - Clone/merge for sub-agent isolation
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_ENTRIES = 100
MAX_TOTAL_BYTES = 25 * 1024 * 1024  # 25 MB


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class FileState:
    """Cached state of a file."""
    content: str
    timestamp: float          # time.monotonic() when cached
    offset: int | None = None       # Partial read offset
    limit: int | None = None        # Partial read limit
    is_partial_view: bool = False    # Auto-injected (CLAUDE.md, etc.) vs full read

    @property
    def size_bytes(self) -> int:
        return len(self.content.encode("utf-8"))


# ── Cache ────────────────────────────────────────────────────────────────────

class FileStateCache:
    """
    LRU cache for file states with size-based eviction.

    All paths are normalized before use as keys.
    Evicts oldest entries when either count or size limit is exceeded.
    """

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ) -> None:
        self._cache: OrderedDict[str, FileState] = OrderedDict()
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._total_bytes = 0

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize a file path for consistent cache keys."""
        return str(Path(path).resolve())

    def get(self, path: str) -> FileState | None:
        """Get cached file state, or None if not cached."""
        key = self._normalize_path(path)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(
        self,
        path: str,
        content: str,
        *,
        offset: int | None = None,
        limit: int | None = None,
        is_partial_view: bool = False,
        timestamp: float | None = None,
    ) -> None:
        """Cache a file's content."""
        import time

        key = self._normalize_path(path)
        ts = timestamp or time.monotonic()

        state = FileState(
            content=content,
            timestamp=ts,
            offset=offset,
            limit=limit,
            is_partial_view=is_partial_view,
        )

        # Remove old entry if exists (to update size tracking)
        if key in self._cache:
            old = self._cache.pop(key)
            self._total_bytes -= old.size_bytes

        # Add new entry
        self._cache[key] = state
        self._total_bytes += state.size_bytes

        # Evict if needed
        self._evict()

    def invalidate(self, path: str) -> None:
        """Remove a file from the cache (e.g., after writing)."""
        key = self._normalize_path(path)
        if key in self._cache:
            old = self._cache.pop(key)
            self._total_bytes -= old.size_bytes
            logger.debug("Invalidated cache for %s", path)

    def has(self, path: str) -> bool:
        """Check if a path is cached."""
        return self._normalize_path(path) in self._cache

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._total_bytes = 0

    def clone(self) -> FileStateCache:
        """Create a deep copy (for sub-agent isolation)."""
        new = FileStateCache(
            max_entries=self._max_entries,
            max_total_bytes=self._max_total_bytes,
        )
        for key, state in self._cache.items():
            new._cache[key] = FileState(
                content=state.content,
                timestamp=state.timestamp,
                offset=state.offset,
                limit=state.limit,
                is_partial_view=state.is_partial_view,
            )
        new._total_bytes = self._total_bytes
        return new

    def merge(self, other: FileStateCache) -> None:
        """Merge another cache into this one (newer timestamps win)."""
        for key, state in other._cache.items():
            existing = self._cache.get(key)
            if existing is None or state.timestamp > existing.timestamp:
                if existing:
                    self._total_bytes -= existing.size_bytes
                self._cache[key] = state
                self._total_bytes += state.size_bytes
        self._evict()

    def get_changed_files(self) -> dict[str, FileState]:
        """Return all cached file states (for diffing)."""
        return dict(self._cache)

    # ── Info ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of cached files."""
        return len(self._cache)

    @property
    def total_bytes(self) -> int:
        """Total bytes of cached content."""
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, path: str) -> bool:
        return self.has(path)

    # ── Internal ─────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Evict oldest entries until within limits."""
        while len(self._cache) > self._max_entries:
            key, state = self._cache.popitem(last=False)
            self._total_bytes -= state.size_bytes
            logger.debug("Evicted %s from file cache (count limit)", key)

        while self._total_bytes > self._max_total_bytes and self._cache:
            key, state = self._cache.popitem(last=False)
            self._total_bytes -= state.size_bytes
            logger.debug("Evicted %s from file cache (size limit)", key)
