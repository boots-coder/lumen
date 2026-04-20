"""
Lumen — Interactive Coding Agent

Run directly and pick a model / enter your API key inside the UI:
    python chat.py

Or specify them on the command line (skipping the wizard):
    python chat.py --model gpt-4o --api-key sk-...

Slash commands:
    /help      Show help
    /status    Token usage and session info
    /mode      Switch general / plan / review (or cycle with Shift+Tab)
    /plan      Plan Mode control (on | approve | cancel)
    /lang      Switch UI language (en | zh)
    /compact   Manually compact context
    /reset     Clear conversation history
    /save      Save session to file
    /load      Load session from file
    /config    Reconfigure model and key
    /forget    Delete locally saved API key
    /quit      Quit

Interaction enhancements:
    • Type / to pop open the command completer
    • Type @ to auto-complete project file paths (30s cache)
    • /load <tab> completes files in the project; picking one loads that session
    • Multi-line input: Enter sends, Shift+Enter / Alt+Enter / Ctrl+J newline
      (Shift+Enter requires a terminal emitting CSI-u or modifyOtherKeys escapes;
       iTerm2/Terminal.app default doesn't distinguish — fall back to Alt+Enter)
    • Input history persisted at ~/.lumen/history, retained across sessions
    • Shift+Tab cycles modes
    • Bottom bar always shows the current mode / phase
    • After first configuration the key is saved in ~/.lumen/config.json (mode 0600),
      reused on next launch — clear with /forget.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle

from lumen import (
    Agent, PermissionChecker, PermissionBehavior, PermissionMode,
    HookRegistry, RetryConfig, Notifier, FileEditTracker,
)
from lumen.i18n import (
    t, tt, set_language, get_language,
    SUPPORTED_LANGUAGES, language_display_name, normalize_language,
)
from lumen.context.transcript import TranscriptReader, list_recent_sessions
from chat_commands import (
    ChatContext, CommandRegistry, CommandResult, SlashCommand,
)
from lumen.services.permissions import PermissionRule
from lumen.services.review_state import ReviewPhase
from lumen.tools import (
    FileReadTool, FileWriteTool, FileEditTool,
    GlobTool, GrepTool, BashTool, TreeTool, DefinitionsTool,
)
from lumen.tools.approval import ConsoleApprovalHandler

console = Console()

# ── Color palette ────────────────────────────────────────────────────────────
C_BRAND   = "bold cyan"
C_USER    = "bold green"
C_AGENT   = "bold blue"
C_DIM     = "dim white"
C_WARN    = "bold yellow"
C_SUCCESS = "bold green"
C_CMD     = "bold magenta"
C_NUM     = "bold cyan"

BANNER = r"""
  ██╗     ██╗   ██╗███╗   ███╗███████╗███╗   ██╗
  ██║     ██║   ██║████╗ ████║██╔════╝████╗  ██║
  ██║     ██║   ██║██╔████╔██║█████╗  ██╔██╗ ██║
  ██║     ██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║
  ███████╗╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║
  ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝
"""

# ── Shared state (accessed by keybinding / bottom_toolbar closures) ──────────
_AGENT_REF: dict = {"agent": None}
_PT_SESSION_REF: dict = {"session": None}      # Main input session (/mode, Shift+Tab)
_INNER_PT_REF: dict = {"session": None}         # Modal input session (permission / approval)
_LIVE_REF: dict = {"live": None}                # Current agent_response Live, for pause/resume


class _LivePause:
    """Context manager that suspends any active Rich Live region while a
    modal prompt runs, then resumes it. Prevents prompt_toolkit from fighting
    Rich for terminal control, which is what makes modal prompts appear to
    'swallow Enter' — they were never visible in the first place."""

    def __enter__(self):
        self._live = _LIVE_REF.get("live")
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                self._live = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
            try:
                self._live.start()
            except Exception:
                pass
        return False


class _LivePausingApprovalHandler:
    """Wraps any ApprovalHandler so each `.request()` call is bracketed by
    a `_LivePause`. Without this, the approval panel draws inside an active
    Rich Live region and prompt_toolkit can't take the terminal — the user
    sees nothing and their Enter appears to do nothing."""

    def __init__(self, inner) -> None:
        self._inner = inner

    async def request(self, phase, title, content):
        with _LivePause():
            return await self._inner.request(phase, title, content)


# ── Slash / @path completion ──────────────────────────────────────────────────
class SlashCompleter(Completer):
    """
    Unified completer, three trigger categories:
      · `/`           Top-level command menu (/help /mode /load …)
      · `/mode r…`    Sub-arguments (general / review)
      · `/load …`     Project file paths (for loading session files)
      · `@path/to/…`  Project file references (inserts text, agent sees the path)

    File-tree scan results cached for 30s, auto-skipping common noise dirs
    (.git / node_modules / __pycache__ / .venv / …) so complete_while_typing
    doesn't walk the disk on every keystroke.
    """

    # Fallback lists used only before the CommandRegistry is attached
    # (e.g. during setup_wizard). After chat_loop builds the registry, it
    # calls `.bind_registry(reg)` and we read live from there.
    @property
    def TOP_LEVEL(self):
        return [
            ("/help",    t("cmd.help")),
            ("/config",  t("cmd.config")),
            ("/quit",    t("cmd.quit")),
        ]

    @property
    def MODE_ARGS(self):
        return [
            ("general", t("cmd.sub.mode.general")),
            ("review",  t("cmd.sub.mode.review")),
        ]

    SKIP_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".tox", ".next", ".turbo", ".cache", "target", ".idea", ".vscode",
    }
    _FILE_CACHE_TTL = 30.0
    _FILE_CACHE_MAX = 2000

    def __init__(self) -> None:
        self._file_cache: list[str] | None = None
        self._file_cache_ts: float = 0.0
        self._registry = None  # type: ignore  # Optional[CommandRegistry]

    def bind_registry(self, registry) -> None:
        """Attach a CommandRegistry so completion reflects user-loaded commands."""
        self._registry = registry

    def _top_level_items(self) -> list[tuple[str, str]]:
        if self._registry is not None:
            return [(c.name, c.description_text()) for c in self._registry.all()]
        return list(self.TOP_LEVEL)

    def _subargs_for(self, head: str) -> list[tuple[str, str]]:
        if self._registry is not None:
            cmd = self._registry.find(head)
            if cmd is not None and cmd.subargs:
                return cmd.subarg_items()
        if head == "/mode":
            return list(self.MODE_ARGS)
        return []

    def _scan_files(self) -> list[str]:
        import time
        now = time.time()
        if (
            self._file_cache is not None
            and (now - self._file_cache_ts) < self._FILE_CACHE_TTL
        ):
            return self._file_cache

        cwd = Path.cwd()
        results: list[str] = []
        for root, dirs, files in os.walk(cwd):
            dirs[:] = [
                d for d in dirs
                if d not in self.SKIP_DIRS and not d.startswith(".")
            ]
            root_path = Path(root)
            for f in files:
                if f.startswith(".") and f not in (".gitignore", ".env.example"):
                    continue
                full = root_path / f
                try:
                    rel = full.relative_to(cwd)
                except ValueError:
                    continue
                results.append(str(rel))
                if len(results) >= self._FILE_CACHE_MAX:
                    break
            if len(results) >= self._FILE_CACHE_MAX:
                break
        results.sort()
        self._file_cache = results
        self._file_cache_ts = now
        return results

    def _rank_files(self, prefix: str, limit: int = 50) -> list[str]:
        """Return files matching prefix, ranked: basename-start > path-start > substring."""
        if not prefix:
            return self._scan_files()[:limit]
        lower = prefix.lower()
        ranked: list[tuple[int, str]] = []
        for path in self._scan_files():
            p_lower = path.lower()
            base_lower = Path(path).name.lower()
            if base_lower.startswith(lower):
                rank = 0
            elif p_lower.startswith(lower):
                rank = 1
            elif lower in p_lower:
                rank = 2
            else:
                continue
            ranked.append((rank, path))
        ranked.sort(key=lambda x: (x[0], x[1]))
        return [p for _, p in ranked[:limit]]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # ── @path file reference (allowed anywhere in a sentence) ────────────
        at_idx = text.rfind("@")
        if at_idx != -1 and (at_idx == 0 or text[at_idx - 1] in " \t\n"):
            prefix = text[at_idx + 1:]
            if prefix == "" or (" " not in prefix and "\t" not in prefix and "\n" not in prefix):
                for path in self._rank_files(prefix):
                    yield Completion(
                        path, start_position=-len(prefix),
                        display=path, display_meta="file",
                    )
                return

        # ── /load <path> file-path completion ────────────────────────────────
        if text.startswith("/load "):
            arg = text[len("/load "):].lstrip()
            for path in self._rank_files(arg):
                yield Completion(
                    path, start_position=-len(arg),
                    display=path, display_meta="session",
                )
            return

        # ── Sub-arguments (e.g. /mode review, any registered subargs) ───────
        if text.startswith("/") and " " in text:
            head, _, rest = text.partition(" ")
            prefix = rest.lstrip()
            subargs = self._subargs_for(head)
            for arg, desc in subargs:
                if arg.startswith(prefix):
                    yield Completion(
                        arg, start_position=-len(prefix),
                        display=arg, display_meta=desc,
                    )
            return

        # ── Top-level slash command ──────────────────────────────────────────
        if text.startswith("/"):
            for cmd, desc in self._top_level_items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd, start_position=-len(text),
                        display=cmd, display_meta=desc,
                    )

# ─────────────────────────────────────────────────────────────────────────────
# Provider / Model catalog
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelOption:
    id: str
    desc: str  # Raw description — may be an i18n key if prefixed with "i18n:"

    def display(self) -> str:
        """Resolve the description, translating if it uses an i18n key."""
        if self.desc.startswith("i18n:"):
            return t(self.desc[5:])
        # Mixed template: leading "<model>  " + suffix i18n key
        if "||" in self.desc:
            prefix, key = self.desc.split("||", 1)
            return prefix + t(key)
        return self.desc

@dataclass
class ProviderOption:
    id: str
    name: str              # Fixed brand name or an i18n key prefixed "i18n:"
    key_hint: str          # API key format hint (may be i18n: key)
    env_key: str | None    # Env var to read the key from
    base_url: str | None   # None = let the Agent auto-detect
    models: list[ModelOption]
    needs_key: bool = True

    def display_name(self) -> str:
        return t(self.name[5:]) if self.name.startswith("i18n:") else self.name

    def display_key_hint(self) -> str:
        return t(self.key_hint[5:]) if self.key_hint.startswith("i18n:") else self.key_hint

PROVIDERS: list[ProviderOption] = [
    ProviderOption(
        id="openai", name="OpenAI", key_hint="sk-proj-...",
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        models=[
            ModelOption("gpt-4.1-2025-04-14", "GPT-4.1         1M ctx · ||provider.model.latest_strongest"),
            ModelOption("gpt-4.1-mini-2025-04-14", "GPT-4.1 Mini    1M ctx · ||provider.model.fast_cheap"),
            ModelOption("gpt-4.1-nano-2025-04-14", "GPT-4.1 Nano    1M ctx · ||provider.model.ultra_cheap"),
            ModelOption("gpt-4o",         "GPT-4o          128K ctx · ||provider.model.allround"),
            ModelOption("gpt-4o-mini",    "GPT-4o Mini     128K ctx · ||provider.model.fast_cheap"),
            ModelOption("o3-mini",        "o3-mini         200K ctx · ||provider.model.reasoning"),
        ],
    ),
    ProviderOption(
        id="getgoapi", name="i18n:provider.getgoapi.name", key_hint="sk-...",
        env_key="GETGOAPI_API_KEY",
        base_url="https://api.getgoapi.com/v1",
        models=[
            ModelOption("gpt-4.1-2025-04-14", "GPT-4.1         ||provider.model.latest_strongest"),
            ModelOption("gpt-4.1-mini-2025-04-14", "GPT-4.1 Mini    ||provider.model.fast_cheap"),
            ModelOption("gpt-4.1-nano-2025-04-14", "GPT-4.1 Nano    ||provider.model.ultra_cheap"),
            ModelOption("gpt-4o",         "GPT-4o          ||provider.model.allround"),
            ModelOption("gpt-4o-mini",    "GPT-4o Mini     ||provider.model.fast_cheap"),
            ModelOption("o3-mini",        "o3-mini         ||provider.model.reasoning"),
            ModelOption("claude-sonnet-4-6", "i18n:provider.model.sonnet_strongest"),
            ModelOption("claude-haiku-4-5",  "i18n:provider.model.haiku_fastcheap"),
            ModelOption("deepseek-chat",  "i18n:provider.model.deepseek_chat"),
            ModelOption("deepseek-reasoner", "i18n:provider.model.deepseek_reasoner"),
        ],
    ),
    ProviderOption(
        id="anthropic", name="Anthropic (Claude)", key_hint="sk-ant-...",
        env_key="ANTHROPIC_API_KEY",
        base_url=None,
        models=[
            ModelOption("claude-sonnet-4-6",         "Claude Sonnet 4.6    200K ctx · ||provider.model.latest_strongest"),
            ModelOption("claude-opus-4-6",            "Claude Opus 4.6      200K ctx · ||provider.model.latest_strongest"),
            ModelOption("claude-haiku-4-5",           "Claude Haiku 4.5     200K ctx · ||provider.model.fast_cheap"),
            ModelOption("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet    200K ctx"),
        ],
    ),
    ProviderOption(
        id="deepseek", name="DeepSeek", key_hint="sk-...",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        models=[
            ModelOption("deepseek-chat",     "DeepSeek Chat      64K ctx · ||provider.model.allround"),
            ModelOption("deepseek-reasoner", "DeepSeek Reasoner  64K ctx · ||provider.model.reasoning"),
        ],
    ),
    ProviderOption(
        id="ollama", name="i18n:provider.ollama.name", key_hint="i18n:provider.ollama.key_hint",
        env_key=None,
        base_url="http://localhost:11434/v1",
        needs_key=False,
        models=[
            ModelOption("llama3.1",      "Llama 3.1     128K ctx"),
            ModelOption("llama3.2",      "Llama 3.2     128K ctx"),
            ModelOption("qwen2.5",       "Qwen 2.5      128K ctx"),
            ModelOption("mistral",       "Mistral        32K ctx"),
            ModelOption("deepseek-r1",   "DeepSeek R1    64K ctx"),
            ModelOption("phi3",          "Phi-3         128K ctx"),
        ],
    ),
    ProviderOption(
        id="custom", name="i18n:provider.custom.name", key_hint="i18n:provider.custom.key_hint",
        env_key=None,
        base_url=None,
        models=[],   # User-supplied
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Configuration wizard
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    api_key: str
    model: str
    base_url: str | None
    provider_name: str
    language: str = "en"
    provider_id: str = ""  # Stable id (e.g. "getgoapi"); enables live re-translation


def display_provider_name(config: AgentConfig) -> str:
    """Return the provider's display name, translated live when the provider
    is known by id. Falls back to whatever was stored in `provider_name`."""
    pid = config.provider_id or _infer_provider_id(config.provider_name)
    if pid:
        for p in PROVIDERS:
            if p.id == pid:
                return p.display_name()
    return config.provider_name


def _infer_provider_id(stored_name: str) -> str:
    """Best-effort recovery of a provider id from a legacy stored display name.
    Matches against the EN + ZH translation of every provider's name."""
    if not stored_name:
        return ""
    from lumen import i18n as _i18n
    for p in PROVIDERS:
        if not p.name.startswith("i18n:"):
            if p.name == stored_name:
                return p.id
            continue
        key = p.name[5:]
        for lang in _i18n.SUPPORTED_LANGUAGES:
            translated = _i18n.TRANSLATIONS.get(lang, {}).get(key)
            if translated == stored_name:
                return p.id
    return ""


# ── Config persistence ───────────────────────────────────────────────────────
# Stored at ~/.lumen/config.json  (file mode 0600) so users don't have to
# re-enter the API key on every launch.

_CONFIG_DIR = Path.home() / ".lumen"
_CONFIG_FILE = _CONFIG_DIR / "config.json"


def save_config(config: AgentConfig) -> None:
    """Write config to ~/.lumen/config.json with 0600 permissions."""
    import json
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "api_key": config.api_key,
            "model": config.model,
            "base_url": config.base_url,
            "provider_name": config.provider_name,
            "provider_id": config.provider_id,
            "language": config.language,
        }
        _CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(_CONFIG_FILE, 0o600)
        except Exception:
            pass
    except Exception as e:
        render_system(t("wizard.save_failed", err=e), C_DIM)


def load_config() -> AgentConfig | None:
    """Load cached config if present and well-formed."""
    import json
    if not _CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if not data.get("model") or not data.get("api_key"):
            return None
        lang = normalize_language(data.get("language")) or get_language()
        return AgentConfig(
            api_key=data["api_key"],
            model=data["model"],
            base_url=data.get("base_url"),
            provider_name=data.get("provider_name") or "Saved",
            provider_id=data.get("provider_id") or "",
            language=lang,
        )
    except Exception:
        return None


def forget_config() -> bool:
    """Delete the saved config file."""
    try:
        if _CONFIG_FILE.exists():
            _CONFIG_FILE.unlink()
            return True
    except Exception:
        pass
    return False


async def setup_wizard(pt: PromptSession) -> AgentConfig:
    """Interactive configuration wizard: pick Provider → pick Model → enter Key."""
    console.print()
    console.print(Rule(t("wizard.title"), style=C_BRAND))
    console.print()

    # ── Step 1: pick Provider ────────────────────────────────────────────────
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style=C_NUM,  width=4)
    table.add_column(style="white")
    table.add_column(style=C_DIM)

    for i, p in enumerate(PROVIDERS, 1):
        key_info = f"env: {p.env_key}" if p.env_key and os.environ.get(p.env_key or "") else ""
        detected = f"  [green]✓ Key detected[/]" if p.env_key and os.environ.get(p.env_key or "") else ""
        table.add_row(f"[{i}]", p.display_name() + detected, key_info)

    console.print(Panel(table, title=t("wizard.pick_provider"), border_style=C_DIM))

    while True:
        try:
            raw = await pt.prompt_async(t("wizard.enter_number"))
            idx = int(raw.strip()) - 1
            if 0 <= idx < len(PROVIDERS):
                provider = PROVIDERS[idx]
                break
            console.print(t("wizard.range_hint", n=len(PROVIDERS)))
        except ValueError:
            console.print(t("wizard.must_be_number"))
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt

    console.print(t("wizard.selected", value=provider.display_name()) + "\n")

    # ── Step 2: pick Model ───────────────────────────────────────────────────
    if provider.id == "custom":
        try:
            base_url = (await pt.prompt_async(t("wizard.custom_base_url"))).strip()
            model    = (await pt.prompt_async(t("wizard.custom_model"))).strip()
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt
        if not model:
            raise ValueError(t("wizard.custom_empty_model"))
    else:
        model_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        model_table.add_column(style=C_NUM, width=4)
        model_table.add_column(style="white")
        for i, m in enumerate(provider.models, 1):
            model_table.add_row(f"[{i}]", m.display())

        console.print(Panel(model_table, title=t("wizard.pick_model"), border_style=C_DIM))

        while True:
            try:
                raw = await pt.prompt_async(t("wizard.enter_number"))
                idx = int(raw.strip()) - 1
                if 0 <= idx < len(provider.models):
                    model = provider.models[idx].id
                    break
                console.print(t("wizard.range_hint", n=len(provider.models)))
            except ValueError:
                console.print(t("wizard.must_be_number"))
            except (KeyboardInterrupt, EOFError):
                raise KeyboardInterrupt

        base_url = provider.base_url

    console.print(t("wizard.selected", value=model) + "\n")

    # ── Step 3: enter API Key ────────────────────────────────────────────────
    if not provider.needs_key:
        api_key = "ollama"
        console.print(t("wizard.local_no_key"))
    else:
        env_val = os.environ.get(provider.env_key or "", "") if provider.env_key else ""
        if env_val:
            masked = env_val[:8] + "..." + env_val[-4:]
            use_env = (await pt.prompt_async(
                t("wizard.key_env_detected", name=provider.env_key, masked=masked)
            )).strip().lower()
            if use_env in ("", "y", "yes"):
                api_key = env_val
                console.print(t("wizard.key_env_used") + "\n")
            else:
                api_key = await _prompt_key(pt, provider.display_key_hint())
        else:
            console.print(t("wizard.key_hint_prefix", hint=provider.display_key_hint()))
            api_key = await _prompt_key(pt, provider.display_key_hint())

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style=C_DIM)
    summary.add_column(style="white")
    summary.add_row(t("col.provider"), provider.display_name())
    summary.add_row(t("col.model"),    model)
    if base_url:
        summary.add_row(t("col.base_url"), base_url)
    key_display = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
    summary.add_row(t("col.api_key"),  key_display)
    console.print(Panel(summary, title=t("wizard.summary_title"), border_style="green"))

    return AgentConfig(
        api_key=api_key,
        model=model,
        base_url=base_url if provider.id != "custom" else (base_url or None),
        provider_name=provider.display_name(),
        provider_id=provider.id,
        language=get_language(),
    )


async def _prompt_key(pt: PromptSession, hint: str) -> str:
    """API Key input with basic masking prompt."""
    while True:
        try:
            key = await pt.prompt_async(t("wizard.key_prompt"))
            key = key.strip()
            if key:
                return key
            console.print(t("wizard.key_empty"))
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt


# ─────────────────────────────────────────────────────────────────────────────
# UI rendering components
# ─────────────────────────────────────────────────────────────────────────────

def make_token_bar(agent: Agent) -> Panel:
    total  = agent.token_usage.total
    window = agent.context_window
    pct    = total / window if window else 0
    bar_style = "green" if pct < 0.6 else ("yellow" if pct < 0.8 else "red")

    progress = Progress(
        TextColumn("[{task.description}]", style=C_DIM),
        BarColumn(bar_width=36, style=bar_style, complete_style=bar_style),
        TextColumn("{task.percentage:>5.1f}%", style=C_DIM),
        expand=False,
    )
    progress.add_task(f"{total:,} / {window:,} tokens", total=window, completed=total)

    stats = Text()
    stats.append(t("tokenbar.model"), style=C_DIM)
    stats.append(agent.model, style=C_BRAND)
    stats.append(t("tokenbar.msgs"), style=C_DIM)
    stats.append(str(len(agent.messages)), style="white")
    stats.append(t("tokenbar.compactions"), style=C_DIM)
    stats.append(str(agent.session.compaction_count), style="white")
    if agent.review_mode:
        stats.append("  │  ", style=C_DIM)
        phase = agent.review_state.phase
        phase_label = {
            ReviewPhase.IDLE: t("tokenbar.review_idle"),
            ReviewPhase.DESIGN: t("tokenbar.review_design"),
            ReviewPhase.PIPELINE: t("tokenbar.review_pipeline"),
            ReviewPhase.IMPL: t("tokenbar.review_impl", n=len(agent.review_state.functions_approved)),
            ReviewPhase.COMPLETE: t("tokenbar.review_complete"),
        }.get(phase, t("tokenbar.review_default"))
        stats.append(f"⚑ {phase_label}", style="bold yellow")
    elif agent.plan_mode:
        stats.append("  │  ", style=C_DIM)
        stats.append(t("tokenbar.plan_mode"), style="bold blue")

    perm_mode = agent.permissions.mode
    if perm_mode != PermissionMode.DEFAULT:
        stats.append("  │  ", style=C_DIM)
        perm_label, perm_style = {
            PermissionMode.ACCEPT_EDITS: ("✎ accept-edits", "bold green"),
            PermissionMode.PLAN:         ("📋 plan-only",    "bold blue"),
            PermissionMode.BYPASS:       ("⚠ bypass",       "bold red"),
        }[perm_mode]
        stats.append(perm_label, style=perm_style)

    border = (
        "red" if perm_mode == PermissionMode.BYPASS
        else "blue" if perm_mode == PermissionMode.PLAN or agent.plan_mode
        else "yellow" if agent.review_mode
        else C_DIM
    )
    return Panel(Columns([progress, stats]), border_style=border, padding=(0, 1))


_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _safe_text(s: str) -> str:
    """Strip lone UTF-16 surrogates that appear when streaming chunks split a
    multi-unit character in half (common with emoji / CJK across chunk
    boundaries). These codepoints are unencodable as UTF-8 and crash the
    console writer."""
    if not s:
        return s
    # Fast path: regex strip of the surrogate range.
    cleaned = _LONE_SURROGATE_RE.sub("", s)
    # Belt-and-braces: round-trip through utf-8 with 'replace' so any
    # remaining unencodable code points become '?' instead of crashing.
    try:
        cleaned.encode("utf-8", "strict")
        return cleaned
    except UnicodeEncodeError:
        return cleaned.encode("utf-8", "replace").decode("utf-8", "replace")


def render_user_message(text_value: str) -> Panel:
    return Panel(
        Text(_safe_text(text_value), style="white"),
        title=Text(t("msg.you"), style=C_USER),
        title_align="left", border_style="green", padding=(0, 1),
    )


def render_system(text_value: str, style: str = C_DIM) -> None:
    console.print(f"  {_safe_text(text_value)}", style=style)


def render_error(text_value: str) -> None:
    console.print(Panel(_safe_text(text_value), border_style="red", title="Error", title_align="left"))


def render_help(registry: "CommandRegistry | None" = None) -> None:
    """Render /help. If a CommandRegistry is supplied, the slash-command rows
    are pulled from it so user-loaded commands show up automatically."""
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column(style=C_CMD)
    table.add_column(style="white")

    if registry is not None:
        for cmd in registry.all():
            hint = cmd.description_text()
            if cmd.aliases:
                hint += f"  (aliases: {' '.join(cmd.aliases)})"
            table.add_row(cmd.name, hint)
    else:
        table.add_row("/help", t("cmd.help"))
        table.add_row("/quit", t("cmd.quit"))

    # Non-slash keybinding reference
    for row in [
        ("",              ""),
        ("↑ / ↓",         t("help.row.history")),
        ("Shift+Enter",   t("help.row.shift_enter")),
        ("Alt+Enter",     t("help.row.alt_enter")),
        ("@path",         t("help.row.at_path")),
        ("Shift+Tab",     t("help.row.shift_tab")),
        ("Ctrl+C",        t("help.row.ctrl_c")),
    ]:
        table.add_row(*row)
    console.print(Panel(table, title=t("help.title"), border_style=C_DIM))


def render_status(agent: Agent) -> None:
    usage = agent.token_usage
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style=C_DIM)
    table.add_column(style="white")
    table.add_row(t("status.model"),          agent.model)
    table.add_row(t("status.context_window"), f"{agent.context_window:,} tokens")
    table.add_row(t("status.tokens_used"),    f"{usage.total:,}  ({usage.total/agent.context_window*100:.1f}%)")
    table.add_row(t("status.messages"),       str(len(agent.messages)))
    table.add_row(t("status.compactions"),    str(agent.session.compaction_count))
    table.add_row(t("status.session_start"),  agent.session.created_at[:19].replace("T", " "))
    console.print(Panel(table, title=t("status.title"), border_style=C_DIM))


# ─────────────────────────────────────────────────────────────────────────────
# Agent response
# ─────────────────────────────────────────────────────────────────────────────

async def agent_response(agent: Agent, message: str) -> str:
    """
    Run a full agent turn: live-display tool calls and render the final
    answer as Markdown.
    """
    tool_log: list[str] = []

    def _tool_panel() -> Panel:
        lines = Text()
        for entry in tool_log[-15:]:
            lines.append(entry + "\n")
        return Panel(
            lines,
            title=Text(t("msg.lumen_thinking"), style=C_AGENT),
            title_align="left",
            border_style="blue",
            padding=(0, 1),
        )

    def on_tool_call(name: str, arguments: dict) -> None:
        arg_hint = ""
        for key in ("path", "file_path", "pattern", "command", "query"):
            if key in arguments:
                val = str(arguments[key])
                arg_hint = f"  [dim]{val[:60]}{'…' if len(val)>60 else ''}[/]"
                break
        # Mark write/edit tools specially
        if name in ("write_file", "edit_file"):
            tool_log.append(f"  [bold yellow]✎ {name}[/]{arg_hint}")
        else:
            tool_log.append(f"  [cyan]⚙ {name}[/]{arg_hint}")

    def on_tool_result(name: str, output: str, is_error: bool) -> None:
        preview = output.strip().splitlines()
        first_line = preview[0][:80] if preview else ""
        more = f"  +{len(preview)-1} lines" if len(preview) > 1 else ""
        status = "[red]✗[/]" if is_error else "[green]✓[/]"
        tool_log.append(f"    {status} {first_line}{more}")

    # ── Use a Live panel to show progress while tools run ────────────────────
    response = None
    if len(agent._tool_registry) > 0:
        with Live(
            _tool_panel(),
            console=console,
            refresh_per_second=8,
            vertical_overflow="ellipsis",
            transient=True,
        ) as live:
            _LIVE_REF["live"] = live
            tool_log.append(t("msg.thinking"))

            async def _chat_with_refresh():
                nonlocal response
                response = await agent.chat(
                    message,
                    on_tool_call=lambda n, a: (on_tool_call(n, a), live.update(_tool_panel())),
                    on_tool_result=lambda n, o, e: (on_tool_result(n, o, e), live.update(_tool_panel())),
                )

            try:
                await _chat_with_refresh()
            finally:
                _LIVE_REF["live"] = None
    else:
        # No tools: stream output straight through
        collected: list[str] = []
        panel_title = Text(t("msg.lumen_streaming"), style=C_AGENT)
        with Live(
            Panel(Text(""), title=panel_title, title_align="left",
                  border_style="blue", padding=(0, 1)),
            console=console,
            refresh_per_second=4,
            vertical_overflow="ellipsis",
            transient=False,
        ) as live:
            _LIVE_REF["live"] = live
            try:
                async for chunk in agent.stream(message):
                    collected.append(chunk)
                    md = Markdown(
                        _safe_text("".join(collected)),
                        code_theme="monokai", inline_code_theme="monokai",
                    )
                    live.update(Panel(
                        md, title=panel_title, title_align="left",
                        border_style="blue", padding=(0, 1),
                        width=min(console.width, 120),
                    ))
            finally:
                _LIVE_REF["live"] = None
        return "".join(collected)

    # ── Render the final answer ──────────────────────────────────────────────
    final_text = _safe_text(response.content if response else "")
    md = Markdown(final_text, code_theme="monokai", inline_code_theme="monokai")
    console.print(Panel(
        md,
        title=Text(t("msg.lumen"), style=C_AGENT),
        title_align="left",
        border_style="blue",
        padding=(0, 1),
        width=min(console.width, 120),
    ))
    return final_text


# ─────────────────────────────────────────────────────────────────────────────
# Mode switching
# ─────────────────────────────────────────────────────────────────────────────

def handle_mode(agent: Agent, arg: str, pt: PromptSession | None = None) -> None:
    arg = arg.strip().lower()

    if not arg:
        if agent.review_mode:
            mode_label = t("mode.review_label")
        elif agent.plan_mode:
            mode_label = t("mode.plan_label")
        else:
            mode_label = t("mode.general_label")
        console.print(Panel(
            t("mode.current", label=mode_label) + "\n\n"
            + t("mode.hint_review") + "\n"
            + t("mode.hint_plan") + "\n"
            + t("mode.hint_general") + "\n"
            + t("mode.hint_shift_tab"),
            border_style=C_DIM, title=t("mode.panel_title"), title_align="left",
        ))
        return

    if arg in ("review", "r", "审阅", "careful"):
        if agent.review_mode:
            render_system(t("mode.already_review"), C_DIM)
        else:
            inner = _INNER_PT_REF.get("session") or pt
            handler = _LivePausingApprovalHandler(
                ConsoleApprovalHandler(pt_session=inner, console=console)
            )
            agent.enable_review_mode(handler=handler)
            console.print(Panel(
                t("mode.review_panel_body"),
                border_style="yellow",
                title=t("mode.review_panel_title"),
                title_align="left",
            ))

    elif arg in ("plan", "p", "方案", "计划"):
        _activate_plan_mode(agent)

    elif arg in ("general", "g", "normal", "通用"):
        was_plan = agent.plan_mode
        was_review = agent.review_mode
        if not was_plan and not was_review:
            render_system(t("mode.already_general"), C_DIM)
        else:
            if was_review:
                agent.disable_review_mode()
            if was_plan:
                agent.disable_plan_mode()
                agent.permissions.set_mode(PermissionMode.DEFAULT)
            render_system(t("mode.back_to_general"), C_SUCCESS)

    else:
        render_system(t("mode.unknown", arg=arg), C_WARN)


def _activate_plan_mode(agent: Agent) -> None:
    if agent.plan_mode:
        render_system(t("mode.already_plan"), C_DIM)
        return
    agent.enable_plan_mode()
    agent.permissions.set_mode(PermissionMode.PLAN)
    console.print(Panel(
        t("mode.plan_panel_body"),
        border_style="blue",
        title=t("mode.plan_panel_title"),
        title_align="left",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Command handling
# ─────────────────────────────────────────────────────────────────────────────

async def handle_compact(agent: Agent) -> None:
    render_system(t("compact.running"), C_WARN)
    try:
        r = await agent.compact(partial=True)
        console.print(Panel(
            t("compact.body",
              mb=r.messages_before, ma=r.messages_after,
              tb=r.tokens_before, ta=r.tokens_after,
              kept=r.kept_recent_count),
            border_style="green", title=t("compact.title"), title_align="left",
        ))
    except Exception as e:
        render_error(t("compact.failed", err=e))


async def handle_save(agent: Agent, pt: PromptSession) -> None:
    default = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        raw = await pt.prompt_async(t("save.prompt", default=default), default=default)
        path = raw.strip() or default
        agent.save_session(path)
        render_system(t("save.done", path=path), C_SUCCESS)
        if agent.transcript:
            render_system(t("save.transcript_hint", path=agent.transcript.path), C_DIM)
    except (KeyboardInterrupt, EOFError):
        render_system(t("common.cancelled"), C_DIM)
    except Exception as e:
        render_error(t("save.failed", err=e))


async def handle_load(
    arg: str, config: AgentConfig, pt: PromptSession
) -> Agent | None:
    if not arg:
        try:
            arg = (await pt.prompt_async(t("load.prompt"))).strip()
        except (KeyboardInterrupt, EOFError):
            return None
    if not arg or not Path(arg).exists():
        render_error(t("load.missing", path=arg))
        return None
    try:
        agent = Agent.load_session(
            arg, api_key=config.api_key,
            model=config.model, base_url=config.base_url,
        )
        render_system(t("load.done", n=len(agent.messages), path=arg), C_SUCCESS)
        return agent
    except Exception as e:
        render_error(t("load.failed", err=e))
        return None


async def handle_resume(config: AgentConfig, pt: PromptSession) -> Agent | None:
    """List recent auto-saved sessions and let the user pick one to resume."""
    cwd = Path.cwd()
    sessions = list_recent_sessions(cwd, limit=10)
    if not sessions:
        render_system(t("resume.none"), C_WARN)
        return None

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("#", style=C_NUM, width=4)
    table.add_column(t("resume.col.session_id"), style="white")
    table.add_column(t("status.model"), style=C_DIM)
    table.add_column(t("resume.col.messages"), style="white", justify="right")
    table.add_column(t("resume.col.created"), style=C_DIM)
    table.add_column(t("resume.col.last_prompt"), style="white")

    for i, s in enumerate(sessions, 1):
        created = s.created_at[:19].replace("T", " ") if s.created_at else "?"
        prompt_preview = s.last_prompt[:50] + ("..." if len(s.last_prompt) > 50 else "")
        table.add_row(
            f"[{i}]",
            s.session_id[:12] + "...",
            s.model or "?",
            str(s.message_count),
            created,
            prompt_preview,
        )

    console.print(Panel(table, title=t("resume.table_title"), border_style=C_DIM))

    try:
        raw = await pt.prompt_async(t("resume.pick_prompt"))
        raw = raw.strip()
        if not raw:
            return None
        idx = int(raw) - 1
        if not (0 <= idx < len(sessions)):
            render_system(t("resume.range_hint", n=len(sessions)), C_WARN)
            return None
    except (ValueError, KeyboardInterrupt, EOFError):
        return None

    chosen = sessions[idx]
    try:
        messages, metadata = TranscriptReader.load_session(chosen.file_path)
        if not messages:
            render_error(t("resume.empty_file"))
            return None

        # Rebuild a Session from the transcript messages
        from lumen.context.session import Session
        session = Session(model=config.model)
        session.session_id = chosen.session_id
        session.messages = messages

        agent = Agent(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            session=session,
            auto_compact=True,
            tools=[
                TreeTool(), DefinitionsTool(), FileReadTool(),
                FileWriteTool(), FileEditTool(),
                GlobTool(), GrepTool(), BashTool(),
            ],
        )
        render_system(
            t("resume.done", sid=chosen.session_id[:12], n=len(messages)),
            C_SUCCESS,
        )
        return agent
    except Exception as e:
        render_error(t("resume.failed", err=e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Slash command registry
# ─────────────────────────────────────────────────────────────────────────────
# All slash commands (/help, /status, /mode, /load, …) register through the
# CommandRegistry. Both SlashCompleter and /help read from here — single
# source of truth. At startup we also auto-scan ~/.lumen/commands/*.py so
# users can add their own commands without editing chat.py.
# ─────────────────────────────────────────────────────────────────────────────

def _build_command_registry() -> CommandRegistry:
    """Build the built-in command registry. Handlers close over ChatContext
    (passed in at dispatch time), so they can see live agent/config/pt state."""
    reg = CommandRegistry()

    async def _quit(ctx, arg):
        render_system(t("common.goodbye"), C_DIM)
        return CommandResult(quit=True)

    async def _help(ctx, arg):
        render_help(ctx.registry)
        return CommandResult()

    async def _status(ctx, arg):
        render_status(ctx.agent)
        return CommandResult()

    async def _mode(ctx, arg):
        handle_mode(ctx.agent, arg, pt=ctx.pt)
        return CommandResult()

    async def _compact(ctx, arg):
        await handle_compact(ctx.agent)
        return CommandResult()

    async def _reset(ctx, arg):
        ctx.agent.reset()
        render_system(t("reset.done"), C_SUCCESS)
        return CommandResult()

    async def _save(ctx, arg):
        await handle_save(ctx.agent, ctx.pt)
        return CommandResult()

    async def _load(ctx, arg):
        new = await handle_load(arg, ctx.config, ctx.pt)
        return CommandResult(new_agent=new) if new else CommandResult()

    async def _resume(ctx, arg):
        new = await handle_resume(ctx.config, ctx.pt)
        return CommandResult(new_agent=new) if new else CommandResult()

    async def _config(ctx, arg):
        render_system(t("config.back_to_wizard"), C_DIM)
        return CommandResult(reconfigure=True)

    async def _forget(ctx, arg):
        if forget_config():
            render_system(t("forget.done", path=_CONFIG_FILE), C_SUCCESS)
        else:
            render_system(t("forget.nothing"), C_DIM)
        return CommandResult()

    async def _lang(ctx, arg):
        """Switch UI language live."""
        arg = arg.strip().lower()
        if not arg:
            current = language_display_name()
            ctx.console.print(Panel(
                t("lang.current", name=current) + "\n\n" + t("lang.available"),
                border_style=C_DIM, title="Language / \u8bed\u8a00",
                title_align="left",
            ))
            return CommandResult()
        target = normalize_language(arg)
        if target is None:
            render_system(t("lang.unknown", arg=arg), C_WARN)
            return CommandResult()
        if target == get_language():
            render_system(t("lang.already", name=language_display_name(target)), C_DIM)
            return CommandResult()
        set_language(target)
        # Persist to config so next launch keeps this choice
        try:
            ctx.config.language = target
            save_config(ctx.config)
        except Exception:
            pass
        render_system(t("lang.switched", name=language_display_name(target)), C_SUCCESS)
        return CommandResult()

    async def _perm(ctx, arg):
        """Switch permission posture. /perm alone shows current + options."""
        checker = ctx.agent.permissions
        current = checker.mode

        labels = {
            PermissionMode.DEFAULT:      tt("perm.label.default"),
            PermissionMode.ACCEPT_EDITS: tt("perm.label.accept_edits"),
            PermissionMode.PLAN:         tt("perm.label.plan"),
            PermissionMode.BYPASS:       tt("perm.label.bypass"),
        }

        arg = arg.strip().lower()
        if not arg:
            lines = [t("perm.current", name=labels[current][0])]
            for m, (name, desc) in labels.items():
                marker = "● " if m == current else "  "
                style = "bold cyan" if m == current else "white"
                lines.append(f"  {marker}[{style}]/perm {name}[/]  {desc}")
            ctx.console.print(Panel(
                "\n".join(lines),
                title=t("perm.panel_title"), title_align="left", border_style=C_DIM,
            ))
            return CommandResult()

        # Parse mode name (accept several aliases — bilingual)
        alias_map = {
            "default": PermissionMode.DEFAULT, "d": PermissionMode.DEFAULT, "常规": PermissionMode.DEFAULT,
            "acceptedits": PermissionMode.ACCEPT_EDITS, "accept": PermissionMode.ACCEPT_EDITS,
            "edits": PermissionMode.ACCEPT_EDITS, "a": PermissionMode.ACCEPT_EDITS, "接受": PermissionMode.ACCEPT_EDITS,
            "plan": PermissionMode.PLAN, "p": PermissionMode.PLAN, "计划": PermissionMode.PLAN,
            "bypass": PermissionMode.BYPASS, "b": PermissionMode.BYPASS, "绕过": PermissionMode.BYPASS,
        }
        target = alias_map.get(arg.replace("-", "").replace("_", ""))
        if target is None:
            render_system(t("perm.unknown", arg=arg), C_WARN)
            return CommandResult()

        if target == current:
            render_system(t("perm.already", name=labels[target][0]), C_DIM)
            return CommandResult()

        if target == PermissionMode.BYPASS:
            # Extra friction for the nuclear option.
            try:
                confirm = await ctx.inner_pt.prompt_async(
                    t("perm.bypass_confirm")
                )
            except (KeyboardInterrupt, EOFError):
                render_system(t("common.cancelled"), C_DIM)
                return CommandResult()
            if confirm.strip().lower() not in ("y", "yes"):
                render_system(t("common.cancelled"), C_DIM)
                return CommandResult()

        checker.set_mode(target)
        render_system(
            t("perm.switched", name=labels[target][0], desc=labels[target][1]),
            C_SUCCESS,
        )
        return CommandResult()

    async def _diff(ctx, arg):
        """Show unified diff of files the agent wrote/edited this session."""
        from rich.syntax import Syntax
        tracker = ctx.edit_tracker
        if tracker is None:
            render_system(t("diff.no_tracker"), C_WARN)
            return CommandResult()

        entries = tracker.entries(dirty_only=True)
        if not entries:
            render_system(t("diff.no_changes"), C_DIM)
            return CommandResult()

        target = arg.strip()
        if target:
            # Filter: user can pass a path (absolute or cwd-relative) or basename
            keep = []
            for e in entries:
                if (
                    e.path == target
                    or Path(e.path).name == target
                    or target in e.path
                ):
                    keep.append(e)
            if not keep:
                render_system(t("diff.no_match", target=target), C_WARN)
                return CommandResult()
            entries = keep

        # Summary table first
        summary = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        summary.add_column(t("diff.col.file"), style="white")
        summary.add_column(t("diff.col.plus"), style="bold green", justify="right")
        summary.add_column(t("diff.col.minus"), style="bold red", justify="right")
        summary.add_column(t("diff.col.writes"), style=C_DIM, justify="right")
        summary.add_column(t("diff.col.status"), style=C_DIM)
        total_added = total_removed = 0
        for e in entries:
            added, removed = e.line_stats()
            total_added += added
            total_removed += removed
            try:
                rel = str(Path(e.path).relative_to(Path.cwd()))
            except ValueError:
                rel = e.path
            summary.add_row(
                rel, f"+{added}", f"-{removed}",
                str(e.write_count),
                t("diff.status_new") if not e.existed_before else t("diff.status_edited"),
            )
        ctx.console.print(Panel(
            summary,
            title=t("diff.panel_title", n=len(entries), added=total_added, removed=total_removed),
            title_align="left", border_style=C_DIM,
        ))

        # Per-file unified diff
        for e in entries:
            diff_text = e.unified_diff()
            if not diff_text:
                continue
            try:
                rel = str(Path(e.path).relative_to(Path.cwd()))
            except ValueError:
                rel = e.path
            ctx.console.print(Panel(
                Syntax(diff_text, "diff", theme="monokai", line_numbers=False,
                       word_wrap=False, background_color="default"),
                title=rel, title_align="left", border_style="blue",
                padding=(0, 1),
            ))

        return CommandResult()

    async def _plan(ctx, arg):
        """Plan Mode control: on/off/approve/cancel. Bare /plan shows status."""
        agent = ctx.agent
        sub = arg.strip().lower()

        if not sub:
            if agent.plan_mode:
                ctx.console.print(Panel(
                    t("plan.active_body"),
                    border_style="blue", title=t("plan.on_title"), title_align="left",
                ))
            else:
                ctx.console.print(Panel(
                    t("plan.inactive_body"),
                    border_style=C_DIM, title=t("plan.on_title"), title_align="left",
                ))
            return CommandResult()

        if sub in ("on", "start", "开启"):
            _activate_plan_mode(agent)
            return CommandResult()

        if sub in ("off", "cancel", "取消"):
            if not agent.plan_mode:
                render_system(t("plan.off_not_active"), C_DIM)
                return CommandResult()
            agent.disable_plan_mode()
            agent.permissions.set_mode(PermissionMode.DEFAULT)
            render_system(t("plan.cancelled"), C_SUCCESS)
            return CommandResult()

        if sub in ("approve", "go", "执行", "ok"):
            if not agent.plan_mode:
                render_system(t("plan.approve_not_active"), C_WARN)
                return CommandResult()
            agent.disable_plan_mode()
            agent.permissions.set_mode(PermissionMode.ACCEPT_EDITS)
            render_system(t("plan.approved"), C_SUCCESS)
            return CommandResult()

        render_system(t("plan.unknown_sub", sub=sub), C_WARN)
        return CommandResult()

    async def _tasks(ctx, arg):
        """Show / inspect / cancel background sub-agent tasks.

        Subcommands:
          /tasks                 → table of all tasks
          /tasks <id>            → show output of that task
          /tasks stop <id>       → cancel
          /tasks clear           → kill every running task + clear finished
        """
        import time as _time
        mgr = ctx.agent.subagent_manager
        sub = arg.strip()
        sub_lower = sub.lower()
        tasks = mgr.list_tasks()

        if sub_lower in ("clear", "reset", "清空"):
            await mgr.cleanup()
            render_system(t("tasks.cleared"), C_SUCCESS)
            return CommandResult()

        if sub_lower.startswith("stop ") or sub_lower.startswith("kill "):
            target = sub.split(maxsplit=1)[1].strip()
            ok = await mgr.kill(target)
            if ok:
                render_system(t("tasks.stopped", id=target), C_SUCCESS)
            else:
                render_system(t("tasks.stop_missing", id=target), C_WARN)
            return CommandResult()

        # /tasks <id>  →  single task detail
        if sub and not sub_lower.startswith(("stop", "kill", "clear")):
            match = next((task for task in tasks if task.agent_id == sub), None)
            if match is None:
                render_system(t("tasks.not_found", id=sub), C_WARN)
                return CommandResult()
            result = await mgr.get_result(sub)
            if result is None:
                ctx.console.print(Panel(
                    f"status: [yellow]{match.status}[/]\n"
                    f"description: {match.description or '(none)'}\n"
                    f"elapsed: {_time.monotonic() - match.start_time:.1f}s",
                    title=f"Task {sub}", title_align="left", border_style=C_DIM,
                ))
                return CommandResult()
            body = result.content if result.success else (result.error or "")
            ctx.console.print(Panel(
                f"[dim]duration {result.duration_ms:.0f}ms · "
                f"tool calls {result.tool_calls_count} · "
                f"success {result.success}[/]\n\n{body}",
                title=f"Task {sub} · {result.description}",
                title_align="left",
                border_style="green" if result.success else "red",
            ))
            return CommandResult()

        # Default: table of all tasks
        if not tasks:
            render_system(t("tasks.empty_hint"), C_DIM)
            return CommandResult()

        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("id", style=C_BRAND)
        table.add_column(t("tasks.col.state"), style="white")
        table.add_column(t("tasks.col.elapsed"), style=C_DIM, justify="right")
        table.add_column(t("tasks.col.description"), style="white")
        now = _time.monotonic()
        status_style = {
            "running":   "[yellow]● running[/]",
            "completed": "[green]✓ completed[/]",
            "failed":    "[red]✗ failed[/]",
            "killed":    "[dim]■ killed[/]",
        }
        for task in tasks:
            elapsed = now - task.start_time if task.start_time else 0.0
            table.add_row(
                task.agent_id,
                status_style.get(task.status, task.status),
                f"{elapsed:.1f}s",
                task.description or "(none)",
            )
        running_n = sum(1 for task in tasks if task.status == "running")
        ctx.console.print(Panel(
            table,
            title=t("tasks.table_title", n=len(tasks), r=running_n),
            title_align="left",
            border_style=C_DIM,
        ))
        render_system(t("tasks.footer_hint"), C_DIM)
        return CommandResult()

    async def _dream(ctx, arg):
        """Inspect / control Auto-Dream (background memory consolidation)."""
        agent = ctx.agent
        service = agent.auto_dream
        sub = arg.strip().lower()

        if service is None:
            render_system(t("dream.disabled_hint"), C_DIM)
            return CommandResult()

        if sub in ("off", "disable", "stop"):
            agent.disable_auto_dream()
            render_system(t("dream.off"), C_SUCCESS)
            return CommandResult()

        if sub in ("force", "now", "立刻"):
            render_system(t("dream.forcing"), C_DIM)
            await service.force_dream()
            stats = service.stats
            render_system(
                t("dream.forced_done", e=stats.extractions, c=stats.consolidations),
                C_SUCCESS,
            )
            return CommandResult()

        if sub in ("consolidate", "sum"):
            render_system(t("dream.consolidating"), C_DIM)
            summary = await service.consolidate_now()
            if summary:
                ctx.console.print(Panel(
                    summary,
                    title=t("dream.consolidate_title"),
                    title_align="left",
                    border_style="magenta",
                ))
            else:
                render_system(t("dream.consolidate_empty"), C_DIM)
            return CommandResult()

        # Default: show stats + config
        stats = service.stats
        cfg = service.config
        mem_count = (
            len(agent.session_memory) if agent.session_memory is not None else 0
        )
        state_tag = (
            t("dream.stats_state_dreaming") if stats.currently_dreaming
            else t("dream.stats_state_idle")
        )
        ctx.console.print(Panel(
            t(
                "dream.stats_body",
                state=state_tag,
                e=stats.extractions,
                c=stats.consolidations,
                last=stats.last_run_at or "[dim]—[/]",
                err=stats.last_error or f"[dim]{t('common.none')}[/]",
                mem=mem_count,
                interval=cfg.interval_turns,
                cons=cfg.consolidation_every,
            ),
            title=t("dream.stats_title"), title_align="left", border_style="magenta",
        ))
        return CommandResult()

    async def _mcp(ctx, arg):
        """MCP server control.

          /mcp                list connected servers and their tools
          /mcp reconnect <n>  drop and re-dial a server
          /mcp disconnect <n> disconnect and unregister tools
        """
        agent = ctx.agent
        mgr = agent.mcp_manager
        sub = arg.strip()
        sub_lower = sub.lower()

        async def _reconnect(name: str) -> None:
            state = mgr.get(name)
            if state is None:
                render_system(t("mcp.no_server", name=name), C_WARN)
                return
            cfg = state.config
            await mgr.disconnect(name)
            new_state = await agent.connect_mcp(cfg)
            if new_state.status == "connected":
                render_system(
                    t("mcp.reconnect_ok", name=name, n=len(new_state.tools)),
                    C_SUCCESS,
                )
            else:
                render_system(
                    t("mcp.reconnect_failed", name=name, err=new_state.error),
                    C_WARN,
                )

        if sub_lower.startswith("reconnect "):
            await _reconnect(sub.split(maxsplit=1)[1].strip())
            return CommandResult()

        if sub_lower.startswith("disconnect "):
            name = sub.split(maxsplit=1)[1].strip()
            ok = await mgr.disconnect(name)
            if ok:
                # Also drop the tools from the registry
                for tname in list(agent._tool_registry.list_tools()):
                    if tname.startswith(f"mcp__{name}__"):
                        agent._tool_registry.unregister(tname)
                render_system(t("mcp.disconnected", name=name), C_SUCCESS)
            else:
                render_system(t("mcp.not_connected", name=name), C_DIM)
            return CommandResult()

        # Default: show table
        servers = mgr.servers()
        if not servers:
            ctx.console.print(Panel(
                t("mcp.empty_body", path=str(_CONFIG_DIR / "mcp.json")),
                border_style=C_DIM, title="MCP", title_align="left",
            ))
            return CommandResult()

        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("server", style=C_BRAND)
        table.add_column(t("mcp.col.status"), style="white")
        table.add_column(t("mcp.col.tools"), justify="right", style=C_DIM)
        table.add_column(t("mcp.col.detail"), style=C_DIM)
        status_mark = {
            "connected":  "[green]● connected[/]",
            "connecting": "[yellow]● connecting[/]",
            "failed":     "[red]✗ failed[/]",
            "closed":     "[dim]■ closed[/]",
            "disabled":   "[dim]⊘ disabled[/]",
            "idle":       "[dim]idle[/]",
        }
        for s in servers:
            cmd_preview = f"{s.config.command} {' '.join(s.config.args[:3])}"
            if len(s.config.args) > 3:
                cmd_preview += " …"
            detail = s.error if s.error else cmd_preview
            table.add_row(
                s.config.name,
                status_mark.get(s.status, s.status),
                str(len(s.tools)),
                detail,
            )
        ctx.console.print(Panel(
            table,
            title=t("mcp.table_title", n=len(servers)),
            title_align="left",
            border_style=C_DIM,
        ))
        # Per-server tool breakdown (compact)
        for s in servers:
            if not s.tools:
                continue
            tool_names = ", ".join(tool.name for tool in s.tools)
            ctx.console.print(
                f"  [dim]{s.config.name}:[/] {tool_names}"
            )
        render_system(t("mcp.footer_hint"), C_DIM)
        return CommandResult()

    async def _vim(ctx, arg):
        from prompt_toolkit.enums import EditingMode
        app = ctx.pt.app
        if app.editing_mode == EditingMode.VI:
            app.editing_mode = EditingMode.EMACS
            render_system(t("vim.off"), C_SUCCESS)
        else:
            app.editing_mode = EditingMode.VI
            render_system(t("vim.on"), C_SUCCESS)
        return CommandResult()

    for cmd in [
        SlashCommand("/help",    lambda: t("cmd.help"),    _help),
        SlashCommand("/status",  lambda: t("cmd.status"),  _status),
        SlashCommand("/mode",    lambda: t("cmd.mode"),    _mode,
                     subargs=(("general", lambda: t("cmd.sub.mode.general")),
                              ("plan",    lambda: t("cmd.sub.mode.plan")),
                              ("review",  lambda: t("cmd.sub.mode.review")))),
        SlashCommand("/plan",    lambda: t("cmd.plan"),    _plan,
                     subargs=(("on",      lambda: t("cmd.sub.plan.on")),
                              ("approve", lambda: t("cmd.sub.plan.approve")),
                              ("cancel",  lambda: t("cmd.sub.plan.cancel")),
                              ("off",     lambda: t("cmd.sub.plan.off")))),
        SlashCommand("/diff",    lambda: t("cmd.diff"),    _diff),
        SlashCommand("/tasks",   lambda: t("cmd.tasks"),   _tasks,
                     subargs=(("clear", lambda: t("cmd.sub.tasks.clear")),
                              ("stop",  lambda: t("cmd.sub.tasks.stop")))),
        SlashCommand("/mcp",     lambda: t("cmd.mcp"),     _mcp,
                     subargs=(("reconnect",  lambda: t("cmd.sub.mcp.reconnect")),
                              ("disconnect", lambda: t("cmd.sub.mcp.disconnect")))),
        SlashCommand("/dream",   lambda: t("cmd.dream"),   _dream,
                     subargs=(("force",       lambda: t("cmd.sub.dream.force")),
                              ("consolidate", lambda: t("cmd.sub.dream.consolidate")),
                              ("off",         lambda: t("cmd.sub.dream.off")))),
        SlashCommand("/perm",    lambda: t("cmd.perm"),    _perm,
                     subargs=(("default",     lambda: t("cmd.sub.perm.default")),
                              ("acceptEdits", lambda: t("cmd.sub.perm.accept_edits")),
                              ("plan",        lambda: t("cmd.sub.perm.plan")),
                              ("bypass",      lambda: t("cmd.sub.perm.bypass")))),
        SlashCommand("/compact", lambda: t("cmd.compact"), _compact),
        SlashCommand("/reset",   lambda: t("cmd.reset"),   _reset),
        SlashCommand("/save",    lambda: t("cmd.save"),    _save),
        SlashCommand("/load",    lambda: t("cmd.load"),    _load),
        SlashCommand("/resume",  lambda: t("cmd.resume"),  _resume),
        SlashCommand("/config",  lambda: t("cmd.config"),  _config),
        SlashCommand("/forget",  lambda: t("cmd.forget"),  _forget),
        SlashCommand("/vim",     lambda: t("cmd.vim"),     _vim),
        SlashCommand("/lang",    lambda: t("cmd.lang"),    _lang,
                     subargs=(("en", lambda: t("cmd.sub.lang.en")),
                              ("zh", lambda: t("cmd.sub.lang.zh")))),
        SlashCommand("/quit",    lambda: t("cmd.quit"),    _quit,
                     aliases=("/exit", "/q")),
    ]:
        reg.register(cmd)

    # Let users drop ~/.lumen/commands/*.py to add their own commands.
    loaded = reg.discover_user_commands(_CONFIG_DIR / "commands")
    if loaded:
        render_system(t("msg.user_cmd_loaded", names=", ".join(loaded)), C_DIM)

    return reg


# ─────────────────────────────────────────────────────────────────────────────
# Keybindings / bottom toolbar
# ─────────────────────────────────────────────────────────────────────────────

def _make_key_bindings() -> KeyBindings:
    """
    Main input-box keybindings:
      · Shift+Tab              cycle general → plan → review → general
      · Enter                  submit the current input (overrides the default
                               newline behaviour under multiline mode)
      · Enter (completion on)  accept the active completion without submitting
      · Shift+Enter / Alt+Enter / Ctrl+J  insert a newline

    About Shift+Enter:
      Terminals by default don't distinguish Shift+Enter from Enter — both
      send \\r. To make Shift+Enter insert a newline the terminal must emit
      one of the escape sequences below (we recognise all of them):
        · ESC [ 13 ; 2 u       (CSI-u / kitty keyboard protocol; on by
                                default in WezTerm/Ghostty/Kitty)
        · ESC [ 27 ; 2 ; 13 ~  (xterm modifyOtherKeys; on by default in
                                xterm/urxvt)
        · ESC \\r               (same as Alt+Enter — iTerm2 users can
                                bind this via Preferences → Profiles →
                                Keys, adding Shift+Return → Send Escape
                                Sequence with an empty payload.)
      On macOS Terminal / iTerm2 default setups Shift+Enter is
      indistinguishable — use Alt+Enter as the fallback.
    """
    kb = KeyBindings()

    @kb.add("s-tab")
    def _toggle_mode(event):
        """Cycle: general → plan → review → general."""
        agent = _AGENT_REF.get("agent")
        inner = _INNER_PT_REF.get("session") or _PT_SESSION_REF.get("session")
        if agent is None:
            return
        if agent.review_mode:
            # review → general
            agent.disable_review_mode()
        elif agent.plan_mode:
            # plan → review
            agent.disable_plan_mode()
            agent.permissions.set_mode(PermissionMode.DEFAULT)
            handler = _LivePausingApprovalHandler(
                ConsoleApprovalHandler(pt_session=inner, console=console)
            )
            agent.enable_review_mode(handler=handler)
        else:
            # general → plan
            agent.enable_plan_mode()
            agent.permissions.set_mode(PermissionMode.PLAN)
        # Force the bottom toolbar to redraw
        event.app.invalidate()

    @kb.add("enter", filter=completion_is_selected)
    def _accept_completion(event):
        """User navigated into the completion menu — Enter accepts, doesn't submit."""
        event.current_buffer.complete_state = None

    @kb.add("enter")
    def _submit(event):
        """Plain Enter submits the line (overrides multiline-mode default newline)."""
        event.current_buffer.validate_and_handle()

    def _insert_newline(event):
        event.current_buffer.newline()

    # Alt+Enter (equivalent to ESC \r) — the main newline key for iTerm2 users
    kb.add("escape", "enter")(_insert_newline)

    # Ctrl+J — fallback when Alt is swallowed by the OS / window manager
    kb.add("c-j")(_insert_newline)

    # Shift+Enter (CSI-u encoding): ESC [ 13 ; 2 u
    # prompt_toolkit parses the ESC prefix as "escape" then matches the
    # remaining bytes one-by-one.
    kb.add("escape", "[", "1", "3", ";", "2", "u")(_insert_newline)

    # Shift+Enter (xterm modifyOtherKeys encoding): ESC [ 27 ; 2 ; 13 ~
    kb.add("escape", "[", "2", "7", ";", "2", ";", "1", "3", "~")(_insert_newline)

    return kb


def _bottom_toolbar():
    """Persistent bottom status bar: mode · phase · keybinding hints."""
    agent = _AGENT_REF.get("agent")
    if agent is None:
        return ""
    if agent.review_mode:
        phase = agent.review_state.phase
        phase_text = {
            ReviewPhase.IDLE: t("toolbar.review_idle"),
            ReviewPhase.DESIGN: t("toolbar.review_design"),
            ReviewPhase.PIPELINE: t("toolbar.review_pipeline"),
            ReviewPhase.IMPL: t("toolbar.review_impl", n=len(agent.review_state.functions_approved)),
            ReviewPhase.COMPLETE: t("toolbar.review_complete"),
        }.get(phase, "")
        mode_part = t("toolbar.mode_review", phase=phase_text)
    elif agent.plan_mode:
        mode_part = t("toolbar.mode_plan")
    else:
        mode_part = t("toolbar.mode_general")
    perm_tag = {
        PermissionMode.DEFAULT:      "",
        PermissionMode.ACCEPT_EDITS: t("toolbar.perm_accept_edits"),
        PermissionMode.PLAN:         t("toolbar.perm_plan"),
        PermissionMode.BYPASS:       t("toolbar.perm_bypass"),
    }[agent.permissions.mode]
    return t("toolbar.line", mode=mode_part, perm=perm_tag, model=agent.model)


# ─────────────────────────────────────────────────────────────────────────────
# Main chat loop
# ─────────────────────────────────────────────────────────────────────────────

async def chat_loop(config: AgentConfig, pt: PromptSession) -> bool:
    """Run the main chat loop. Returns True if the user asked to reconfigure, False to exit."""

    # ── Permission system: ask user for write/risky operations ────────────
    async def ask_permission(tool_name: str, tool_input: dict) -> bool:
        """Interactive permission prompt for write/risky tools.

        Pauses any active Rich Live region so prompt_toolkit can own the
        terminal cleanly, and uses a dedicated modal PromptSession (no
        toolbar / completer / Shift+Tab binding) so keystrokes never get
        intercepted by the main input box's UX helpers.
        """
        arg_hint = ""
        for key in ("file_path", "command", "pattern"):
            if key in tool_input:
                val = _safe_text(str(tool_input[key]))
                arg_hint = f"  {val[:80]}{'…' if len(val)>80 else ''}"
                break

        modal_pt = _INNER_PT_REF.get("session") or pt
        with _LivePause():
            console.print()
            console.print(Panel(
                _safe_text(t("permission.panel_body", tool=tool_name, hint=arg_hint)),
                title=t("permission.panel_title"), title_align="left",
                border_style="yellow",
            ))
            try:
                answer = await modal_pt.prompt_async(t("permission.prompt"))
                return answer.strip().lower() in ("", "y", "yes")
            except (KeyboardInterrupt, EOFError):
                return False

    permission_checker = PermissionChecker(ask_fn=ask_permission)

    try:
        agent = Agent(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            inject_git_state=True,
            inject_memory=True,
            auto_compact=True,
            permission_checker=permission_checker,
            retry_config=RetryConfig(max_retries=10, fallback_model=None),
            enable_file_cache=True,
            tools=[
                TreeTool(),         # project directory tree
                DefinitionsTool(),  # extract class/function definitions from a file
                FileReadTool(),     # read file contents by line range
                FileWriteTool(),    # create / overwrite files
                FileEditTool(),     # exact find-and-replace editing
                GlobTool(),         # match file paths by pattern
                GrepTool(),         # search inside file contents
                BashTool(),         # run shell commands
            ],
        )
    except Exception as e:
        render_error(t("msg.agent_failed", err=e))
        return True

    # Register sub_agent + task_* tools so the model can spawn and observe
    # background work, and `/tasks` has something to show.
    agent.enable_subagents()

    # Discovery utilities: tool_search (rank tools by query) + brief (session summary).
    agent.enable_discovery_tools()

    # Auto-Dream: background memory consolidation. Runs async after every
    # N turns and periodically merges entries into narrative summaries.
    # Disable with LUMEN_NO_DREAM=1 if you want the old inline behavior.
    if os.environ.get("LUMEN_NO_DREAM") != "1":
        agent.enable_auto_dream()

    # Auto-connect MCP servers declared in ~/.lumen/mcp.json (if any).
    mcp_cfg_path = _CONFIG_DIR / "mcp.json"
    if mcp_cfg_path.exists():
        try:
            mcp_states = await agent.connect_mcp_from_config(mcp_cfg_path)
            if mcp_states:
                ok = [s for s in mcp_states if s.status == "connected"]
                bad = [s for s in mcp_states if s.status == "failed"]
                if ok:
                    names = ", ".join(
                        f"{s.config.name}({len(s.tools)})" for s in ok
                    )
                    render_system(t("startup.mcp_connected", names=names), C_DIM)
                for s in bad:
                    render_system(
                        t("startup.mcp_failed", name=s.config.name, err=s.error),
                        C_WARN,
                    )
        except Exception as e:
            render_system(t("startup.mcp_load_failed", err=e), C_WARN)

    _AGENT_REF["agent"] = agent

    # Desktop notifier — fires on turns that exceed LUMEN_NOTIFY_THRESHOLD (15s default).
    # Disable entirely with LUMEN_NOTIFY=0.
    notifier = Notifier()

    # File edit tracker — session-scoped snapshot of files on first write/edit,
    # powers `/diff`.
    edit_tracker = FileEditTracker()
    edit_tracker.install(agent.hooks)

    # Build the slash command registry (scans ~/.lumen/commands/*.py for user cmds).
    command_registry = _build_command_registry()
    # Let the completer see registered commands (inc. user-loaded ones).
    completer = pt.completer
    if hasattr(completer, "bind_registry"):
        completer.bind_registry(command_registry)

    # Modal prompt session (for permission / approval / wizard sub-prompts).
    # Set up in main_async(); here we just reach for it.
    inner_pt = _INNER_PT_REF.get("session") or pt

    # Startup info
    info = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    info.add_column(style=C_DIM)
    info.add_column(style="white")
    info.add_row(t("col.provider"),        display_provider_name(config))
    info.add_row(t("col.model"),           agent.model)
    info.add_row(t("col.session_id"),      agent.session.session_id[:12] + "...")
    info.add_row(t("col.context_window"),  f"{agent.context_window:,} tokens")
    info.add_row(t("col.auto_compact"),    t("startup.auto_compact"))
    if agent.transcript:
        info.add_row(t("col.auto_save"),   t("startup.auto_save", path=agent.transcript.path))
    info.add_row(t("col.git_state"),       t("startup.git_state"))
    info.add_row(t("col.memory_files"),    t("startup.memory_files"))
    info.add_row(t("col.tools_read"),      t("startup.tools_read"))
    info.add_row(t("col.tools_write"),     t("startup.tools_write"))
    info.add_row(t("col.permissions"),     t("startup.permissions"))
    info.add_row(t("col.retry"),           t("startup.retry"))
    info.add_row(t("col.file_cache"),      t("startup.file_cache"))
    info.add_row(
        t("col.desktop_notify"),
        t("startup.notifier_on", channel=notifier.channel, sec=int(notifier.default_threshold_s))
        if notifier.enabled else t("startup.notifier_off"),
    )
    info.add_row(t("col.language"),        language_display_name())
    console.print(Panel(info, border_style=C_DIM, title=t("startup.ready"), title_align="left"))
    console.print(Rule(style=C_DIM))
    console.print(t("startup.hint"))

    _last_ctrl_c = False

    try:
        while True:
            _last_ctrl_c = False
            try:
                console.print(make_token_bar(agent))
                user_input = (await pt.prompt_async("  You › ")).strip()
                if not user_input:
                    continue
            except EOFError:
                console.print()
                render_system(t("common.goodbye"), C_DIM)
                return False
            except KeyboardInterrupt:
                console.print()
                render_system(t("common.goodbye"), C_DIM)
                return False

            # ── Slash commands (dispatched via CommandRegistry) ──────────────
            if user_input.startswith("/"):
                head, _, rest = user_input.partition(" ")
                command = command_registry.find(head.lower())
                if command is None:
                    render_system(t("msg.unknown_command", cmd=user_input), C_WARN)
                    continue
                ctx = ChatContext(
                    agent=agent, config=config, console=console,
                    pt=pt, inner_pt=inner_pt, notifier=notifier,
                    registry=command_registry, edit_tracker=edit_tracker,
                )
                try:
                    result = await command.handler(ctx, rest.strip())
                except (KeyboardInterrupt, EOFError):
                    render_system(t("common.cancelled"), C_DIM)
                    continue
                except Exception as e:
                    render_error(t("msg.cmd_failed", cmd=command.name, err=e))
                    if os.environ.get("LUMEN_DEBUG"):
                        traceback.print_exc()
                    continue
                if result.new_agent is not None:
                    agent = result.new_agent
                    _AGENT_REF["agent"] = agent
                if result.reconfigure:
                    return True
                if result.quit:
                    return False
                continue

            # ── Normal chat turn ──────────────────────────────────────────────
            console.print(render_user_message(user_input))
            console.print()

            import time as _time
            _turn_start = _time.monotonic()
            try:
                agent.reset_abort()  # Clear any previous abort state
                await agent_response(agent, user_input)
                _elapsed = _time.monotonic() - _turn_start
                notifier.notify_if_slow(
                    "Lumen",
                    t("msg.reply_done", preview=user_input[:40]),
                    elapsed_s=_elapsed,
                )
                console.print()
            except KeyboardInterrupt:
                agent.abort("User pressed Ctrl+C")
                if _last_ctrl_c:
                    console.print()
                    render_system(t("common.goodbye"), C_DIM)
                    return False
                _last_ctrl_c = True
                render_system(t("msg.cancel_reply"), C_WARN)
            except Exception as e:
                render_error(str(e))
                if os.environ.get("LUMEN_DEBUG"):
                    traceback.print_exc()
    finally:
        # Flush and close transcript on any exit path
        try:
            await agent.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(
    cli_api_key: str | None,
    cli_model: str | None,
    cli_base_url: str | None,
) -> None:
    # Try to apply saved language preference BEFORE we render anything.
    try:
        _pre = load_config()
        if _pre is not None:
            _lang = normalize_language(_pre.language)
            if _lang:
                set_language(_lang)
    except Exception:
        pass

    # Banner
    console.print(Text(BANNER, style=C_BRAND), justify="center")
    console.print(Align.center(Text(t("banner.tagline"), style=C_DIM)))
    console.print(Align.center(Text(t("banner.subtitle"), style="dim cyan")))
    console.print()

    # Ensure config dir exists before FileHistory tries to open/append to it.
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Vim mode: opt-in via LUMEN_VIM=1 or toggled at runtime via /vim
    vi_mode = os.environ.get("LUMEN_VIM", "").strip() in ("1", "true", "yes", "on")

    pt = PromptSession(
        history=FileHistory(str(_CONFIG_DIR / "history")),
        style=PTStyle.from_dict({
            "prompt": "ansicyan bold",
            "bottom-toolbar": "bg:#1a1a1a #888888",
            "completion-menu.completion": "bg:#1a1a1a #cccccc",
            "completion-menu.completion.current": "bg:#005f87 #ffffff bold",
            "completion-menu.meta.completion": "bg:#1a1a1a #707070 italic",
            "completion-menu.meta.completion.current": "bg:#005f87 #cfcfcf italic",
        }),
        completer=SlashCompleter(),
        complete_while_typing=True,
        key_bindings=_make_key_bindings(),
        bottom_toolbar=_bottom_toolbar,
        refresh_interval=0.5,
        multiline=True,  # Enter=submit, Alt+Enter/Ctrl+J=newline (see _make_key_bindings)
        vi_mode=vi_mode,
    )
    _PT_SESSION_REF["session"] = pt

    # Inner prompt session — bare & separate, for modal prompts (permissions,
    # approvals). No toolbar, no completer, no Shift+Tab binding. This mirrors
    # src/'s pattern where permission dialogs use a dedicated Select component
    # with its own isolated input handling, so no keybindings leak in from
    # the main input box and no background refresh timer redraws over the prompt.
    inner_pt = PromptSession(
        history=None,  # modal answers shouldn't pollute history
        style=PTStyle.from_dict({"prompt": "ansiyellow bold"}),
    )
    _INNER_PT_REF["session"] = inner_pt

    # If CLI already passed enough args, skip the wizard.
    if cli_api_key and cli_model:
        config = AgentConfig(
            api_key=cli_api_key,
            model=cli_model,
            base_url=cli_base_url,
            provider_name="CLI",
            language=get_language(),
        )
        await chat_loop(config, pt)
        return

    # Otherwise: config → chat → (maybe reconfigure) loop
    config: AgentConfig | None = None
    skip_saved_once = False  # True on /config reconfigure: skip the "use saved?" prompt

    while True:
        # Prefer the saved config unless the user actively ran /config
        saved = None if skip_saved_once else load_config()
        skip_saved_once = False

        if saved is not None:
            # Apply saved language preference immediately.
            saved_lang = normalize_language(saved.language)
            if saved_lang and saved_lang != get_language():
                set_language(saved_lang)
            masked_key = (saved.api_key[:8] + "..." + saved.api_key[-4:]
                          if len(saved.api_key) > 12 else "****")
            info = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            info.add_column(style=C_DIM)
            info.add_column(style="white")
            info.add_row(t("col.provider"), display_provider_name(saved))
            info.add_row(t("col.model"),    saved.model)
            if saved.base_url:
                info.add_row(t("col.base_url"), saved.base_url)
            info.add_row(t("col.api_key"),  masked_key)
            info.add_row(t("col.language"), language_display_name())
            console.print(Panel(
                info, border_style="green",
                title=t("wizard.saved_detected"), title_align="left",
            ))
            try:
                raw = await pt.prompt_async(t("wizard.use_saved_prompt"))
            except (KeyboardInterrupt, EOFError):
                console.print()
                render_system(t("wizard.cancel_exit"), C_DIM)
                return
            choice = raw.strip().lower()
            if choice in ("", "y", "yes"):
                config = saved
            elif choice == "f":
                forget_config()
                render_system(t("wizard.forgot"), C_SUCCESS)
                config = None
            else:
                config = None  # fall through to wizard

        if config is None:
            try:
                config = await setup_wizard(pt)
                save_config(config)
                render_system(
                    t("wizard.saved_to", path=_CONFIG_FILE),
                    C_DIM,
                )
            except (KeyboardInterrupt, EOFError):
                console.print()
                render_system(t("wizard.cancel_exit"), C_DIM)
                return

        console.print()
        reconfigure = await chat_loop(config, pt)
        if not reconfigure:
            break
        console.print()
        console.print(Rule(t("wizard.reconfigure"), style=C_BRAND))
        config = None
        skip_saved_once = True


def main() -> None:
    # Make stdout/stderr surrogate-safe so lone UTF-16 surrogates produced by
    # streaming chunk splits never crash the terminal writer. Replaces any
    # unencodable code point with '?' at the final write instead of raising
    # UnicodeEncodeError from deep inside rich/Live/Panel render paths.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    import argparse
    parser = argparse.ArgumentParser(description=t("cli.description"))
    parser.add_argument("--model",    default=None, help=t("cli.arg.model"))
    parser.add_argument("--api-key",  default=None, help=t("cli.arg.api_key"))
    parser.add_argument("--base-url", default=None, help=t("cli.arg.base_url"))
    parser.add_argument(
        "--lang", default=None, choices=list(SUPPORTED_LANGUAGES),
        help=t("cli.arg.lang"),
    )
    args = parser.parse_args()
    if args.lang:
        set_language(args.lang)

    try:
        asyncio.run(main_async(
            cli_api_key=args.api_key,
            cli_model=args.model,
            cli_base_url=args.base_url,
        ))
    except KeyboardInterrupt:
        console.print("\n  [dim]Goodbye![/]")


if __name__ == "__main__":
    main()
