"""
Abort/cancellation — asyncio-based abort controller with child hierarchy.

Mirrors src/utils/abortController.ts:
  - Parent → child signal propagation (one-directional)
  - Child abort does NOT propagate to parent
  - WeakRef-like cleanup via weak sets
  - Sibling abort for bash tool errors
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Callable

logger = logging.getLogger(__name__)


class AbortController:
    """
    Cancellation controller with parent-child hierarchy.

    When a parent is aborted, all children are also aborted.
    When a child is aborted, the parent is NOT affected.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._children: weakref.WeakSet[AbortController] = weakref.WeakSet()
        self._callbacks: list[Callable[[], None]] = []
        self._reason: str | None = None

    @property
    def is_aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def event(self) -> asyncio.Event:
        """The underlying asyncio.Event — can be passed to functions that check it."""
        return self._event

    def abort(self, reason: str = "Aborted") -> None:
        """
        Signal abort. Propagates to all children.
        """
        if self._event.is_set():
            return

        self._reason = reason
        self._event.set()
        logger.debug("AbortController aborted: %s", reason)

        # Propagate to children
        for child in list(self._children):
            try:
                child.abort(reason=f"Parent aborted: {reason}")
            except Exception:
                pass

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass

    def create_child(self) -> AbortController:
        """
        Create a child controller.

        The child will be aborted when this parent is aborted.
        The parent will NOT be aborted when the child is aborted.
        """
        child = AbortController()
        self._children.add(child)

        # If parent is already aborted, abort child immediately
        if self._event.is_set():
            child.abort(reason=f"Parent already aborted: {self._reason}")

        return child

    def create_sibling_abort(self) -> AbortController:
        """
        Create a sibling abort controller.

        Used for bash tool errors — aborts sibling subprocesses
        without aborting the parent query.
        """
        return self.create_child()

    def on_abort(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when aborted."""
        if self._event.is_set():
            callback()
        else:
            self._callbacks.append(callback)

    def check(self) -> None:
        """Raise AbortError if aborted."""
        if self._event.is_set():
            raise AbortError(self._reason or "Aborted")

    def reset(self) -> None:
        """Reset the controller (for reuse)."""
        self._event.clear()
        self._reason = None
        self._callbacks.clear()
        self._children = weakref.WeakSet()


class AbortError(Exception):
    """Raised when an operation is aborted."""

    def __init__(self, reason: str = "Aborted") -> None:
        self.reason = reason
        super().__init__(reason)
