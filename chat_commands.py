"""
Slash command registry — extensible dispatcher for chat.py.

Built-in commands (/help, /status, /mode, /load, …) register here, and
`~/.lumen/commands/*.py` gets auto-discovered at startup so users can drop
their own commands in without touching chat.py.

Each user command file must expose a module-level `COMMAND: SlashCommand`.

    # ~/.lumen/commands/greet.py
    from chat_commands import SlashCommand, CommandResult

    async def _handle(ctx, arg):
        ctx.console.print(f"hello {arg or 'world'}")
        return CommandResult()

    COMMAND = SlashCommand(
        name="/greet",
        description="打招呼",
        handler=_handle,
    )

The registry is also the single source of truth for:
  · SlashCompleter's top-level completion menu
  · /help's rendered table
so adding a command updates all three automatically.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable


@dataclass
class CommandResult:
    """Return value a handler can use to influence the main loop."""
    quit: bool = False
    reconfigure: bool = False
    new_agent: Any | None = None   # Replace loop's `agent` (from /load, /resume)


@dataclass
class ChatContext:
    """Bundle of live state handed to every command handler."""
    agent: Any              # lumen.Agent
    config: Any             # AgentConfig from chat.py
    console: Any            # rich.Console
    pt: Any                 # main PromptSession
    inner_pt: Any           # modal PromptSession
    notifier: Any           # lumen.Notifier
    registry: Any = None    # CommandRegistry (for /help et al.)
    edit_tracker: Any = None  # FileEditTracker (for /diff)


CommandHandler = Callable[[ChatContext, str], Awaitable[CommandResult]]


@dataclass
class SlashCommand:
    name: str                                    # "/help" (include the slash)
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()                # e.g. ("/q", "/exit")
    subargs: tuple[tuple[str, str], ...] = ()    # (arg, meta) for completion

    def matches(self, token: str) -> bool:
        return token == self.name or token in self.aliases


class CommandRegistry:
    """Ordered registry. First-registered wins on name collision (so user
    commands can't silently shadow built-ins unless explicitly overridden)."""

    def __init__(self) -> None:
        self._commands: list[SlashCommand] = []
        self._by_token: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand, *, override: bool = False) -> None:
        if cmd.name in self._by_token and not override:
            return
        if override and cmd.name in self._by_token:
            old = self._by_token[cmd.name]
            self._commands = [c for c in self._commands if c is not old]
        self._commands.append(cmd)
        self._by_token[cmd.name] = cmd
        for alias in cmd.aliases:
            self._by_token.setdefault(alias, cmd)

    def find(self, token: str) -> SlashCommand | None:
        return self._by_token.get(token)

    def all(self) -> list[SlashCommand]:
        return list(self._commands)

    def discover_user_commands(self, user_dir: Path) -> list[str]:
        """Scan `user_dir` for `*.py` files exporting a `COMMAND` object.
        Returns the list of command names loaded. Broken files are skipped
        with a warning rather than failing startup."""
        loaded: list[str] = []
        if not user_dir.is_dir():
            return loaded
        for py_file in sorted(user_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"lumen_user_cmd_{py_file.stem}", py_file
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                cmd = getattr(mod, "COMMAND", None)
                if isinstance(cmd, SlashCommand):
                    self.register(cmd, override=True)
                    loaded.append(cmd.name)
            except Exception as e:
                print(f"  ⚠ 用户命令 {py_file.name} 加载失败: {e}")
        return loaded
