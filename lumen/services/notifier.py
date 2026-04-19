"""
Desktop notification service — fire-and-forget alerts for long-running tasks.

Usage:

    from lumen.services.notifier import Notifier

    notifier = Notifier()  # auto-detects platform
    notifier.notify("Lumen", "Build finished in 12.3s")

    # Or only notify if the current run took longer than a threshold
    notifier.notify_if_slow("Lumen", "chat done", elapsed_s=18.0, threshold_s=15.0)

Channels tried in order, first-match wins:
  · macOS:  osascript `display notification`
  · Linux:  notify-send
  · Fallback everywhere: terminal bell (\\a)

Every channel is optional — if none are available, notify() is a no-op.
Notifications are dispatched in a background thread so they never block the
main agent loop.

Configurable via env vars:
  · LUMEN_NOTIFY=0           → disable entirely
  · LUMEN_NOTIFY_BELL=0      → disable the terminal-bell fallback
  · LUMEN_NOTIFY_THRESHOLD=N → default seconds threshold for notify_if_slow
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class Notifier:
    """Cross-platform desktop notifier. Thread-safe, non-blocking."""

    def __init__(
        self,
        enabled: bool | None = None,
        bell: bool | None = None,
        threshold_s: float | None = None,
    ) -> None:
        if enabled is None:
            enabled = os.environ.get("LUMEN_NOTIFY", "1").strip().lower() not in ("0", "false", "no", "off")
        if bell is None:
            bell = os.environ.get("LUMEN_NOTIFY_BELL", "1").strip().lower() not in ("0", "false", "no", "off")
        if threshold_s is None:
            try:
                threshold_s = float(os.environ.get("LUMEN_NOTIFY_THRESHOLD", "15"))
            except ValueError:
                threshold_s = 15.0

        self.enabled = enabled
        self.bell = bell
        self.default_threshold_s = threshold_s
        self._channel = self._detect_channel() if enabled else None

    @staticmethod
    def _detect_channel() -> str | None:
        if sys.platform == "darwin" and shutil.which("osascript"):
            return "osascript"
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            return "notify-send"
        return None

    # ── public API ───────────────────────────────────────────────────────────

    def notify(self, title: str, message: str) -> None:
        """Fire a desktop notification. Non-blocking; swallows all errors."""
        if not self.enabled:
            return

        # Dispatch the actual channel call in a daemon thread so we never
        # block the caller even if osascript hangs for a second.
        threading.Thread(
            target=self._send,
            args=(title, message),
            daemon=True,
        ).start()

    def notify_if_slow(
        self,
        title: str,
        message: str,
        elapsed_s: float,
        threshold_s: float | None = None,
    ) -> None:
        """Only notify if elapsed_s exceeded threshold (default from env)."""
        cutoff = threshold_s if threshold_s is not None else self.default_threshold_s
        if elapsed_s >= cutoff:
            self.notify(title, f"{message}  ({elapsed_s:.1f}s)")

    @property
    def channel(self) -> str:
        """Human-readable description of the active channel (for /status UI)."""
        if not self.enabled:
            return "disabled"
        parts: list[str] = []
        if self._channel:
            parts.append(self._channel)
        if self.bell:
            parts.append("bell")
        return " + ".join(parts) if parts else "none (no supported backend)"

    # ── internals ────────────────────────────────────────────────────────────

    def _send(self, title: str, message: str) -> None:
        if self._channel == "osascript":
            self._send_osascript(title, message)
        elif self._channel == "notify-send":
            self._send_linux(title, message)
        if self.bell:
            self._send_bell()

    @staticmethod
    def _send_osascript(title: str, message: str) -> None:
        # Escape " and \ for AppleScript string literal
        esc_title = title.replace("\\", "\\\\").replace('"', '\\"')
        esc_msg = message.replace("\\", "\\\\").replace('"', '\\"')
        script = f'display notification "{esc_msg}" with title "{esc_title}"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False, timeout=3,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.debug("osascript notify failed: %s", e)

    @staticmethod
    def _send_linux(title: str, message: str) -> None:
        try:
            subprocess.run(
                ["notify-send", "--app-name=Lumen", title, message],
                check=False, timeout=3,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.debug("notify-send failed: %s", e)

    @staticmethod
    def _send_bell() -> None:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    # Quick manual test: `python -m lumen.services.notifier`
    n = Notifier()
    print(f"channel = {n.channel}")
    n.notify("Lumen test", "Hello from the notifier.")
    n.notify_if_slow("Lumen test", "this one is slow", elapsed_s=20.0, threshold_s=5.0)
    n.notify_if_slow("Lumen test", "this one is fast (won't fire)", elapsed_s=1.0, threshold_s=5.0)
    import time; time.sleep(1)
    print("done")
