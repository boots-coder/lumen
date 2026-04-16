"""
Lumen — Interactive Coding Agent

直接运行，在 UI 内选择模型和输入 API Key：
    python chat.py

也支持命令行直接指定（跳过配置向导）：
    python chat.py --model gpt-4o --api-key sk-...

聊天中的 Slash 命令：
    /help      帮助
    /status    Token 用量和会话信息
    /mode      切换 general ↔ review（或 Shift+Tab）
    /compact   手动压缩上下文
    /reset     清空对话历史
    /save      保存会话到文件
    /load      从文件加载会话
    /config    重新配置模型和 Key
    /forget    删除本地保存的 API Key
    /quit      退出

交互增强：
    • 输入 / 自动弹出命令补全
    • 输入 @ 自动补全项目文件路径（30s 缓存）
    • /load <tab> 补全项目内文件，选中会话文件加载
    • 多行输入: Enter 发送, Shift+Enter / Alt+Enter / Ctrl+J 换行
      (Shift+Enter 要求终端发出 CSI-u 或 modifyOtherKeys 转义序列;
       iTerm2/Terminal.app 默认不区分, 用 Alt+Enter 兜底即可)
    • 输入历史持久化到 ~/.lumen/history，跨会话保留
    • Shift+Tab 在模式间快速切换
    • 底栏常驻显示当前 mode / phase
    • 首次配置后 Key 会保存在 ~/.lumen/config.json (权限 0600),
      下次启动直接复用 — 用 /forget 清除。
"""

from __future__ import annotations

import asyncio
import os
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

from lumen import Agent, PermissionChecker, PermissionBehavior, HookRegistry, RetryConfig
from lumen.context.transcript import TranscriptReader, list_recent_sessions
from lumen.services.permissions import PermissionRule
from lumen.services.review_state import ReviewPhase
from lumen.tools import (
    FileReadTool, FileWriteTool, FileEditTool,
    GlobTool, GrepTool, BashTool, TreeTool, DefinitionsTool,
)
from lumen.tools.approval import ConsoleApprovalHandler

console = Console()

# ── 配色 ──────────────────────────────────────────────────────────────────────
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

# ── 共享状态（供 keybinding / bottom_toolbar 闭包访问） ──────────────────────────
_AGENT_REF: dict = {"agent": None}
_PT_SESSION_REF: dict = {"session": None}      # 主输入会话（/mode, Shift+Tab）
_INNER_PT_REF: dict = {"session": None}         # 模态输入会话（permission / approval）
_LIVE_REF: dict = {"live": None}                # 当前 agent_response 的 Live，用于暂停/恢复


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


# ── Slash / @path 补全 ────────────────────────────────────────────────────────
class SlashCompleter(Completer):
    """
    统一补全器，支持三类触发：
      · `/`           顶层命令菜单（/help /mode /load …）
      · `/mode r…`    子参数（general / review）
      · `/load …`     项目内文件路径（用于加载会话文件）
      · `@path/to/…`  项目文件引用（插入文本，agent 能看到路径）

    文件树扫描结果缓存 30s，自动跳过常见噪声目录（.git / node_modules /
    __pycache__ / .venv …），防止 complete_while_typing 每次按键都遍历磁盘。
    """

    TOP_LEVEL = [
        ("/help",    "显示帮助"),
        ("/status",  "Token 用量和会话信息"),
        ("/mode",    "切换模式 (general | review)"),
        ("/compact", "手动压缩上下文"),
        ("/reset",   "清空对话历史"),
        ("/save",    "保存会话到文件"),
        ("/load",    "从文件加载会话"),
        ("/resume",  "恢复最近的自动保存会话"),
        ("/config",  "重新配置模型和 Key"),
        ("/forget",  "删除本地保存的 API Key"),
        ("/quit",    "退出"),
    ]
    MODE_ARGS = [
        ("general", "通用模式"),
        ("review",  "审阅模式（阶段门控）"),
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

        # ── @path 文件引用（允许嵌在句子任意位置） ─────────────────────────────
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

        # ── /load <path> 文件路径补全 ────────────────────────────────────────
        if text.startswith("/load "):
            arg = text[len("/load "):].lstrip()
            for path in self._rank_files(arg):
                yield Completion(
                    path, start_position=-len(arg),
                    display=path, display_meta="session",
                )
            return

        # ── /mode 子参数 ─────────────────────────────────────────────────────
        if text.startswith("/mode "):
            prefix = text[len("/mode "):].lstrip()
            for arg, desc in self.MODE_ARGS:
                if arg.startswith(prefix):
                    yield Completion(
                        arg, start_position=-len(prefix),
                        display=arg, display_meta=desc,
                    )
            return

        # ── 顶层 slash 命令 ──────────────────────────────────────────────────
        if text.startswith("/"):
            for cmd, desc in self.TOP_LEVEL:
                if cmd.startswith(text):
                    yield Completion(
                        cmd, start_position=-len(text),
                        display=cmd, display_meta=desc,
                    )

# ─────────────────────────────────────────────────────────────────────────────
# Provider / Model 目录
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelOption:
    id: str
    desc: str

@dataclass
class ProviderOption:
    id: str
    name: str
    key_hint: str          # API key 格式提示
    env_key: str | None    # 从哪个环境变量读取 key
    base_url: str | None   # None = 由 Agent 自动判断
    models: list[ModelOption]
    needs_key: bool = True

PROVIDERS: list[ProviderOption] = [
    ProviderOption(
        id="openai", name="OpenAI", key_hint="sk-proj-...",
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        models=[
            ModelOption("gpt-4.1-2025-04-14", "GPT-4.1         1M ctx · 最新最强"),
            ModelOption("gpt-4.1-mini-2025-04-14", "GPT-4.1 Mini    1M ctx · 快速省钱"),
            ModelOption("gpt-4.1-nano-2025-04-14", "GPT-4.1 Nano    1M ctx · 极速超省"),
            ModelOption("gpt-4o",         "GPT-4o          128K ctx · 综合能力"),
            ModelOption("gpt-4o-mini",    "GPT-4o Mini     128K ctx · 快速省钱"),
            ModelOption("o3-mini",        "o3-mini         200K ctx · 推理模型"),
        ],
    ),
    ProviderOption(
        id="getgoapi", name="GetGoAPI (多模型代理)", key_hint="sk-...",
        env_key="GETGOAPI_API_KEY",
        base_url="https://api.getgoapi.com/v1",
        models=[
            ModelOption("gpt-4.1-2025-04-14", "GPT-4.1         最新最强综合"),
            ModelOption("gpt-4.1-mini-2025-04-14", "GPT-4.1 Mini    快速省钱"),
            ModelOption("gpt-4.1-nano-2025-04-14", "GPT-4.1 Nano    极速超省"),
            ModelOption("gpt-4o",         "GPT-4o          综合能力"),
            ModelOption("gpt-4o-mini",    "GPT-4o Mini     快速省钱"),
            ModelOption("o3-mini",        "o3-mini         推理模型"),
            ModelOption("claude-sonnet-4-6", "Claude Sonnet 4.6  最强综合"),
            ModelOption("claude-haiku-4-5",  "Claude Haiku 4.5   快速省钱"),
            ModelOption("deepseek-chat",  "DeepSeek Chat   通用对话"),
            ModelOption("deepseek-reasoner", "DeepSeek Reasoner 推理模型"),
        ],
    ),
    ProviderOption(
        id="anthropic", name="Anthropic (Claude)", key_hint="sk-ant-...",
        env_key="ANTHROPIC_API_KEY",
        base_url=None,
        models=[
            ModelOption("claude-sonnet-4-6",         "Claude Sonnet 4.6    200K ctx · 最强综合"),
            ModelOption("claude-opus-4-6",            "Claude Opus 4.6      200K ctx · 最强能力"),
            ModelOption("claude-haiku-4-5",           "Claude Haiku 4.5     200K ctx · 快速省钱"),
            ModelOption("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet    200K ctx"),
        ],
    ),
    ProviderOption(
        id="deepseek", name="DeepSeek", key_hint="sk-...",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        models=[
            ModelOption("deepseek-chat",     "DeepSeek Chat      64K ctx · 通用对话"),
            ModelOption("deepseek-reasoner", "DeepSeek Reasoner  64K ctx · 推理模型"),
        ],
    ),
    ProviderOption(
        id="ollama", name="Ollama (本地模型)", key_hint="无需 Key",
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
        id="custom", name="自定义 / 其他", key_hint="你的 API Key",
        env_key=None,
        base_url=None,
        models=[],   # 用户自己输入
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 配置向导
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    api_key: str
    model: str
    base_url: str | None
    provider_name: str


# ── 配置持久化 ────────────────────────────────────────────────────────────────
# 保存在 ~/.lumen/config.json  (文件权限 0600)
# 避免每次启动都要重新粘贴 API Key。

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
        }
        _CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(_CONFIG_FILE, 0o600)
        except Exception:
            pass
    except Exception as e:
        render_system(f"(无法保存配置: {e})", C_DIM)


def load_config() -> AgentConfig | None:
    """Load cached config if present and well-formed."""
    import json
    if not _CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        if not data.get("model") or not data.get("api_key"):
            return None
        return AgentConfig(
            api_key=data["api_key"],
            model=data["model"],
            base_url=data.get("base_url"),
            provider_name=data.get("provider_name") or "Saved",
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
    """交互式配置向导：选 Provider → 选模型 → 输入 Key。"""
    console.print()
    console.print(Rule("  配置模型", style=C_BRAND))
    console.print()

    # ── 第一步：选 Provider ───────────────────────────────────────────────────
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style=C_NUM,  width=4)
    table.add_column(style="white")
    table.add_column(style=C_DIM)

    for i, p in enumerate(PROVIDERS, 1):
        key_info = f"env: {p.env_key}" if p.env_key and os.environ.get(p.env_key or "") else ""
        detected = f"  [green]✓ Key 已检测[/]" if p.env_key and os.environ.get(p.env_key or "") else ""
        table.add_row(f"[{i}]", p.name + detected, key_info)

    console.print(Panel(table, title="选择 Provider", border_style=C_DIM))

    while True:
        try:
            raw = await pt.prompt_async("  输入编号 › ")
            idx = int(raw.strip()) - 1
            if 0 <= idx < len(PROVIDERS):
                provider = PROVIDERS[idx]
                break
            console.print(f"  [yellow]请输入 1–{len(PROVIDERS)} 之间的数字[/]")
        except ValueError:
            console.print("  [yellow]请输入数字[/]")
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt

    console.print(f"  [green]✓ 已选：[/]{provider.name}\n")

    # ── 第二步：选模型 ────────────────────────────────────────────────────────
    if provider.id == "custom":
        try:
            base_url = (await pt.prompt_async("  Base URL (如 http://localhost:11434/v1) › ")).strip()
            model    = (await pt.prompt_async("  Model 名称 › ")).strip()
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt
        if not model:
            raise ValueError("model 不能为空")
    else:
        model_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        model_table.add_column(style=C_NUM, width=4)
        model_table.add_column(style="white")
        for i, m in enumerate(provider.models, 1):
            model_table.add_row(f"[{i}]", m.desc)

        console.print(Panel(model_table, title="选择模型", border_style=C_DIM))

        while True:
            try:
                raw = await pt.prompt_async("  输入编号 › ")
                idx = int(raw.strip()) - 1
                if 0 <= idx < len(provider.models):
                    model = provider.models[idx].id
                    break
                console.print(f"  [yellow]请输入 1–{len(provider.models)} 之间的数字[/]")
            except ValueError:
                console.print("  [yellow]请输入数字[/]")
            except (KeyboardInterrupt, EOFError):
                raise KeyboardInterrupt

        base_url = provider.base_url

    console.print(f"  [green]✓ 已选：[/]{model}\n")

    # ── 第三步：输入 API Key ──────────────────────────────────────────────────
    if not provider.needs_key:
        api_key = "ollama"
        console.print(f"  [dim]本地模型无需 API Key，使用默认占位符[/]\n")
    else:
        env_val = os.environ.get(provider.env_key or "", "") if provider.env_key else ""
        if env_val:
            masked = env_val[:8] + "..." + env_val[-4:]
            use_env = (await pt.prompt_async(
                f"  检测到环境变量 {provider.env_key} ({masked})\n"
                f"  直接使用? [Y/n] › "
            )).strip().lower()
            if use_env in ("", "y", "yes"):
                api_key = env_val
                console.print("  [green]✓ 使用环境变量中的 Key[/]\n")
            else:
                api_key = await _prompt_key(pt, provider.key_hint)
        else:
            console.print(f"  [dim]格式参考：{provider.key_hint}[/]")
            api_key = await _prompt_key(pt, provider.key_hint)

    # ── 汇总展示 ──────────────────────────────────────────────────────────────
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style=C_DIM)
    summary.add_column(style="white")
    summary.add_row("Provider", provider.name)
    summary.add_row("Model",    model)
    if base_url:
        summary.add_row("Base URL", base_url)
    key_display = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
    summary.add_row("API Key",  key_display)
    console.print(Panel(summary, title="✓ 配置确认", border_style="green"))

    return AgentConfig(
        api_key=api_key,
        model=model,
        base_url=base_url if provider.id != "custom" else (base_url or None),
        provider_name=provider.name,
    )


async def _prompt_key(pt: PromptSession, hint: str) -> str:
    """带密码掩码的 API Key 输入。"""
    while True:
        try:
            key = await pt.prompt_async("  API Key › ")
            key = key.strip()
            if key:
                return key
            console.print("  [yellow]Key 不能为空，请重新输入[/]")
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt


# ─────────────────────────────────────────────────────────────────────────────
# UI 渲染组件
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
    stats.append("  model: ", style=C_DIM)
    stats.append(agent.model, style=C_BRAND)
    stats.append("  │  msgs: ", style=C_DIM)
    stats.append(str(len(agent.messages)), style="white")
    stats.append("  │  compactions: ", style=C_DIM)
    stats.append(str(agent.session.compaction_count), style="white")
    if agent.review_mode:
        stats.append("  │  ", style=C_DIM)
        phase = agent.review_state.phase
        phase_label = {
            ReviewPhase.IDLE: "审阅·待命",
            ReviewPhase.DESIGN: "审阅·设计",
            ReviewPhase.PIPELINE: "审阅·数据流",
            ReviewPhase.IMPL: f"审阅·实现({len(agent.review_state.functions_approved)})",
            ReviewPhase.COMPLETE: "审阅·完成",
        }.get(phase, "审阅")
        stats.append(f"⚑ {phase_label}", style="bold yellow")

    border = "yellow" if agent.review_mode else C_DIM
    return Panel(Columns([progress, stats]), border_style=border, padding=(0, 1))


def render_user_message(text: str) -> Panel:
    return Panel(
        Text(text, style="white"),
        title=Text("  You", style=C_USER),
        title_align="left", border_style="green", padding=(0, 1),
    )


def render_system(text: str, style: str = C_DIM) -> None:
    console.print(f"  {text}", style=style)


def render_error(text: str) -> None:
    console.print(Panel(text, border_style="red", title="Error", title_align="left"))


def render_help() -> None:
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column(style=C_CMD)
    table.add_column(style="white")
    for cmd, desc in [
        ("/help",         "显示帮助"),
        ("/status",       "Token 用量和会话信息"),
        ("/mode",         "切换模式：通用 ↔ 审阅  (/mode review | general,  或 Shift+Tab 快速切换)"),
        ("/compact",      "手动压缩上下文"),
        ("/reset",        "清空对话历史"),
        ("/save",         "保存会话到文件"),
        ("/load",         "从文件加载会话"),
        ("/resume",       "恢复最近的自动保存会话"),
        ("/config",       "重新配置模型和 Key"),
        ("/forget",       "删除本地保存的 API Key"),
        ("/quit",         "退出  (或 Ctrl+C / Ctrl+D)"),
        ("",              ""),
        ("↑ / ↓",         "浏览输入历史（持久化在 ~/.lumen/history）"),
        ("Shift+Enter",   "换行（需终端支持；iTerm2 用户可改用 Alt+Enter）"),
        ("Alt+Enter",     "换行（跨终端兜底，Ctrl+J 同效）"),
        ("@path",         "自动补全项目内文件路径"),
        ("Ctrl+C",        "取消当前输入 / 再按一次退出"),
    ]:
        table.add_row(cmd, desc)
    console.print(Panel(table, title="命令列表", border_style=C_DIM))


def render_status(agent: Agent) -> None:
    usage = agent.token_usage
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style=C_DIM)
    table.add_column(style="white")
    table.add_row("Model",          agent.model)
    table.add_row("Context window", f"{agent.context_window:,} tokens")
    table.add_row("Tokens used",    f"{usage.total:,}  ({usage.total/agent.context_window*100:.1f}%)")
    table.add_row("Messages",       str(len(agent.messages)))
    table.add_row("Compactions",    str(agent.session.compaction_count))
    table.add_row("Session start",  agent.session.created_at[:19].replace("T", " "))
    console.print(Panel(table, title="Session Status", border_style=C_DIM))


# ─────────────────────────────────────────────────────────────────────────────
# Agent 响应
# ─────────────────────────────────────────────────────────────────────────────

async def agent_response(agent: Agent, message: str) -> str:
    """
    执行一次完整的 Agent 响应：工具调用过程实时展示，最终答案 Markdown 渲染。
    """
    tool_log: list[str] = []

    def _tool_panel() -> Panel:
        lines = Text()
        for entry in tool_log[-15:]:
            lines.append(entry + "\n")
        return Panel(
            lines,
            title=Text("  Lumen  ●  thinking…", style=C_AGENT),
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

    # ── 有工具时用 Live 面板展示进度 ──────────────────────────────────────────
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
            tool_log.append("  [dim]正在思考…[/]")

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
        # 无工具时走流式输出
        collected: list[str] = []
        panel_title = Text("  Lumen  ●  streaming…", style=C_AGENT)
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
                    md = Markdown("".join(collected), code_theme="monokai", inline_code_theme="monokai")
                    live.update(Panel(
                        md, title=panel_title, title_align="left",
                        border_style="blue", padding=(0, 1),
                        width=min(console.width, 120),
                    ))
            finally:
                _LIVE_REF["live"] = None
        return "".join(collected)

    # ── 渲染最终答案 ───────────────────────────────────────────────────────────
    final_text = response.content if response else ""
    md = Markdown(final_text, code_theme="monokai", inline_code_theme="monokai")
    console.print(Panel(
        md,
        title=Text("  Lumen", style=C_AGENT),
        title_align="left",
        border_style="blue",
        padding=(0, 1),
        width=min(console.width, 120),
    ))
    return final_text


# ─────────────────────────────────────────────────────────────────────────────
# 模式切换
# ─────────────────────────────────────────────────────────────────────────────

def handle_mode(agent: Agent, arg: str, pt: PromptSession | None = None) -> None:
    arg = arg.strip().lower()

    if not arg:
        mode = "审阅模式  [Review Mode]" if agent.review_mode else "通用模式  [General]"
        console.print(Panel(
            f"  当前模式: [bold cyan]{mode}[/]\n\n"
            "  [dim]/mode review[/]   → 开启审阅模式（阶段门控）\n"
            "  [dim]/mode general[/]  → 切回通用模式\n"
            "  [dim]Shift+Tab[/]       → 快速切换",
            border_style=C_DIM, title="模式", title_align="left",
        ))
        return

    if arg in ("review", "r", "审阅", "careful"):
        if agent.review_mode:
            render_system("已经是审阅模式。", C_DIM)
        else:
            inner = _INNER_PT_REF.get("session") or pt
            handler = _LivePausingApprovalHandler(
                ConsoleApprovalHandler(pt_session=inner, console=console)
            )
            agent.enable_review_mode(handler=handler)
            console.print(Panel(
                "[bold yellow]审阅模式已开启 — 真正的阶段门控[/]\n\n"
                "  模型写代码时分 3 步进行，每步 AI 会[bold]调用 request_approval 工具[/]暂停等你确认:\n"
                "  · Phase 1  [cyan]设计方案[/] — 展示函数/类签名和职责\n"
                "  · Phase 2  [blue]数据流[/]   — 文本箭头图展示 pipeline\n"
                "  · Phase 3  [yellow]逐步实现[/] — 一次一个函数，解释数据变化\n\n"
                "  审批面板弹出时你可以:\n"
                "  · [bold]Y / 空回车[/] → 通过，进入下一步\n"
                "  · [bold]skip / 全部写完[/] → 通过并跳过后续审批\n"
                "  · [bold]N[/] 然后输入反馈 → 打回重写\n"
                "  · 直接输入任何文字 → 当作反馈并打回\n\n"
                "  [dim]Token 栏会实时显示当前 Phase。非编码任务不受影响。[/]",
                border_style="yellow", title="⚑ Review Mode", title_align="left",
            ))

    elif arg in ("general", "g", "normal", "通用"):
        if not agent.review_mode:
            render_system("已经是通用模式。", C_DIM)
        else:
            agent.disable_review_mode()
            render_system("✓ 已切回通用模式。", C_SUCCESS)

    else:
        render_system(
            f"未知模式 '{arg}'。用 /mode review | /mode general。", C_WARN
        )


# ─────────────────────────────────────────────────────────────────────────────
# 命令处理
# ─────────────────────────────────────────────────────────────────────────────

async def handle_compact(agent: Agent) -> None:
    render_system("Compacting context…", C_WARN)
    try:
        r = await agent.compact(partial=True)
        console.print(Panel(
            f"[green]✓ 压缩完成[/]\n\n"
            f"  消息数 : [white]{r.messages_before}[/] → [white]{r.messages_after}[/]\n"
            f"  Tokens : [white]{r.tokens_before:,}[/] → [white]{r.tokens_after:,}[/]\n"
            f"  保留最近 [white]{r.kept_recent_count}[/] 条消息原文",
            border_style="green", title="压缩结果", title_align="left",
        ))
    except Exception as e:
        render_error(f"压缩失败: {e}")


async def handle_save(agent: Agent, pt: PromptSession) -> None:
    default = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        raw = await pt.prompt_async(f"  保存到 [{default}] › ", default=default)
        path = raw.strip() or default
        agent.save_session(path)
        render_system(f"✓ 已保存 → {path}", C_SUCCESS)
        if agent.transcript:
            render_system(
                f"  (会话也在自动保存到 {agent.transcript.path})", C_DIM,
            )
    except (KeyboardInterrupt, EOFError):
        render_system("已取消", C_DIM)
    except Exception as e:
        render_error(f"保存失败: {e}")


async def handle_load(
    arg: str, config: AgentConfig, pt: PromptSession
) -> Agent | None:
    if not arg:
        try:
            arg = (await pt.prompt_async("  加载文件路径 › ")).strip()
        except (KeyboardInterrupt, EOFError):
            return None
    if not arg or not Path(arg).exists():
        render_error(f"文件不存在: {arg!r}")
        return None
    try:
        agent = Agent.load_session(
            arg, api_key=config.api_key,
            model=config.model, base_url=config.base_url,
        )
        render_system(f"✓ 已加载 {len(agent.messages)} 条消息 from {arg}", C_SUCCESS)
        return agent
    except Exception as e:
        render_error(f"加载失败: {e}")
        return None


async def handle_resume(config: AgentConfig, pt: PromptSession) -> Agent | None:
    """List recent auto-saved sessions and let the user pick one to resume."""
    cwd = Path.cwd()
    sessions = list_recent_sessions(cwd, limit=10)
    if not sessions:
        render_system("没有找到自动保存的会话。", C_WARN)
        return None

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("#", style=C_NUM, width=4)
    table.add_column("Session ID", style="white")
    table.add_column("Model", style=C_DIM)
    table.add_column("Messages", style="white", justify="right")
    table.add_column("Created", style=C_DIM)
    table.add_column("Last prompt", style="white")

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

    console.print(Panel(table, title="最近的自动保存会话", border_style=C_DIM))

    try:
        raw = await pt.prompt_async("  输入编号恢复 (或 Enter 取消) › ")
        raw = raw.strip()
        if not raw:
            return None
        idx = int(raw) - 1
        if not (0 <= idx < len(sessions)):
            render_system(f"请输入 1-{len(sessions)} 之间的数字", C_WARN)
            return None
    except (ValueError, KeyboardInterrupt, EOFError):
        return None

    chosen = sessions[idx]
    try:
        messages, metadata = TranscriptReader.load_session(chosen.file_path)
        if not messages:
            render_error("会话文件为空。")
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
            f"✓ 已恢复会话 {chosen.session_id[:12]}... "
            f"({len(messages)} 条消息)",
            C_SUCCESS,
        )
        return agent
    except Exception as e:
        render_error(f"恢复失败: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 快捷键 / 底栏
# ─────────────────────────────────────────────────────────────────────────────

def _make_key_bindings() -> KeyBindings:
    """
    主输入框键位:
      · Shift+Tab              general ↔ review 模式切换
      · Enter                  提交当前输入（在 multiline 模式下覆盖默认换行）
      · Enter (补全选中时)     接受当前补全项，不提交
      · Shift+Enter / Alt+Enter / Ctrl+J  插入换行

    关于 Shift+Enter:
      终端默认不区分 Shift+Enter 和 Enter — 两者都发 \\r。要让 Shift+Enter 换行，
      终端必须发出以下任一转义序列之一（下面我们都识别）:
        · ESC [ 13 ; 2 u   (CSI-u / kitty keyboard protocol, WezTerm/Ghostty/Kitty 默认开)
        · ESC [ 27 ; 2 ; 13 ~   (xterm modifyOtherKeys, xterm/urxvt 默认开)
        · ESC \\r               (等同于 Alt+Enter —— iTerm2 用户可以在
                                Preferences → Profiles → Keys 新增快捷键
                                Shift+Return → Send Escape Sequence → 留空即可)
      在 macOS Terminal / iTerm2 默认设置下 Shift+Enter 不区分，用 Alt+Enter 兜底。
    """
    kb = KeyBindings()

    @kb.add("s-tab")
    def _toggle_mode(event):
        agent = _AGENT_REF.get("agent")
        inner = _INNER_PT_REF.get("session") or _PT_SESSION_REF.get("session")
        if agent is None:
            return
        if agent.review_mode:
            agent.disable_review_mode()
        else:
            handler = _LivePausingApprovalHandler(
                ConsoleApprovalHandler(pt_session=inner, console=console)
            )
            agent.enable_review_mode(handler=handler)
        # 强制重绘底栏
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

    # Alt+Enter（等同于 ESC \r）—— iTerm2 用户的主力换行键
    kb.add("escape", "enter")(_insert_newline)

    # Ctrl+J —— 在 Alt 被系统/窗口管理器吞掉时的兜底
    kb.add("c-j")(_insert_newline)

    # Shift+Enter (CSI-u 编码): ESC [ 13 ; 2 u
    # prompt_toolkit 把 ESC 前缀解析为 "escape", 然后逐字符匹配余下字节
    kb.add("escape", "[", "1", "3", ";", "2", "u")(_insert_newline)

    # Shift+Enter (xterm modifyOtherKeys 编码): ESC [ 27 ; 2 ; 13 ~
    kb.add("escape", "[", "2", "7", ";", "2", ";", "1", "3", "~")(_insert_newline)

    return kb


def _bottom_toolbar():
    """底部常驻状态条：mode · phase · 快捷键提示。"""
    agent = _AGENT_REF.get("agent")
    if agent is None:
        return ""
    if agent.review_mode:
        phase = agent.review_state.phase
        phase_text = {
            ReviewPhase.IDLE: "待命",
            ReviewPhase.DESIGN: "设计",
            ReviewPhase.PIPELINE: "数据流",
            ReviewPhase.IMPL: f"实现×{len(agent.review_state.functions_approved)}",
            ReviewPhase.COMPLETE: "完成",
        }.get(phase, "")
        mode_part = f" ⚑ review · {phase_text} "
    else:
        mode_part = " general "
    return (
        f" mode:{mode_part}│ ⇧⇥ 切模式 │ / 命令 │ @ 文件 │ "
        f"⇧⏎ / ⌥⏎ 换行 │ {agent.model} "
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主聊天循环
# ─────────────────────────────────────────────────────────────────────────────

async def chat_loop(config: AgentConfig, pt: PromptSession) -> bool:
    """运行聊天主循环。返回 True 表示重新配置，False 表示退出。"""

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
                val = str(tool_input[key])
                arg_hint = f"  {val[:80]}{'…' if len(val)>80 else ''}"
                break

        modal_pt = _INNER_PT_REF.get("session") or pt
        with _LivePause():
            console.print()
            console.print(Panel(
                f"  工具: [bold yellow]{tool_name}[/]\n{arg_hint}",
                title="🔒 需要确认", title_align="left", border_style="yellow",
            ))
            try:
                answer = await modal_pt.prompt_async("  允许执行? [Y/n] › ")
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
                TreeTool(),         # 项目目录结构树
                DefinitionsTool(),  # 提取文件中所有类/函数定义
                FileReadTool(),     # 按行读取文件内容
                FileWriteTool(),    # 创建/覆写文件
                FileEditTool(),     # 精确查找替换编辑
                GlobTool(),         # 按模式匹配文件路径
                GrepTool(),         # 搜索文件内容
                BashTool(),         # 执行 shell 命令
            ],
        )
    except Exception as e:
        render_error(f"Agent 创建失败: {e}")
        return True

    _AGENT_REF["agent"] = agent

    # 启动信息
    info = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    info.add_column(style=C_DIM)
    info.add_column(style="white")
    info.add_row("Provider",       config.provider_name)
    info.add_row("Model",          agent.model)
    info.add_row("Session ID",     agent.session.session_id[:12] + "...")
    info.add_row("Context window", f"{agent.context_window:,} tokens")
    info.add_row("Auto-compact",   "✓ 开启")
    if agent.transcript:
        info.add_row("Auto-save",  f"✓ {agent.transcript.path}")
    info.add_row("Git state",      "✓ 注入")
    info.add_row("Memory files",   "✓ ENGRAM.md 自动发现")
    info.add_row("Tools (读)",     "✓ tree · definitions · read_file · glob · grep")
    info.add_row("Tools (写)",     "[yellow]✎[/] write_file · edit_file · bash")
    info.add_row("权限系统",       "✓ 读操作静默放行 · 写操作需确认 · 危险命令拒绝")
    info.add_row("重试/降级",      "✓ 指数退避 · 429/529重试 · max_tokens自动缩减")
    info.add_row("文件缓存",       "✓ LRU 100文件/25MB")
    console.print(Panel(info, border_style=C_DIM, title="✓ 准备就绪", title_align="left"))
    console.print(Rule(style=C_DIM))
    console.print(
        "  输入消息后按 Enter 发送。"
        "  [magenta]/help[/] 查看命令。"
        "  [dim]Ctrl+D 或 /quit 退出。[/]\n"
    )

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
                render_system("Goodbye!", C_DIM)
                return False
            except KeyboardInterrupt:
                console.print()
                render_system("Goodbye!", C_DIM)
                return False

            cmd = user_input.lower()

            # ── Slash 命令 ────────────────────────────────────────────────────
            if cmd in ("/quit", "/exit", "/q"):
                render_system("Goodbye!", C_DIM)
                return False

            elif cmd == "/help":
                render_help();  continue

            elif cmd == "/status":
                render_status(agent);  continue

            elif cmd.startswith("/mode"):
                handle_mode(agent, user_input[5:].strip(), pt=pt);  continue

            elif cmd == "/compact":
                await handle_compact(agent);  continue

            elif cmd == "/reset":
                agent.reset()
                render_system("✓ 对话历史已清空。", C_SUCCESS);  continue

            elif cmd == "/save":
                await handle_save(agent, pt);  continue

            elif cmd.startswith("/load"):
                new = await handle_load(user_input[5:].strip(), config, pt)
                if new:
                    agent = new
                    _AGENT_REF["agent"] = agent
                continue

            elif cmd == "/resume":
                new = await handle_resume(config, pt)
                if new:
                    agent = new
                    _AGENT_REF["agent"] = agent
                continue

            elif cmd == "/config":
                render_system("返回配置向导…", C_DIM)
                return True

            elif cmd == "/forget":
                if forget_config():
                    render_system(
                        f"✓ 已删除本地保存的配置 ({_CONFIG_FILE})", C_SUCCESS,
                    )
                else:
                    render_system("没有保存的配置需要删除。", C_DIM)
                continue

            elif cmd.startswith("/"):
                render_system(f"未知命令: {user_input}  输入 /help 查看帮助", C_WARN)
                continue

            # ── 正常聊天 ──────────────────────────────────────────────────────
            console.print(render_user_message(user_input))
            console.print()

            try:
                agent.reset_abort()  # Clear any previous abort state
                await agent_response(agent, user_input)
                console.print()
            except KeyboardInterrupt:
                agent.abort("User pressed Ctrl+C")
                if _last_ctrl_c:
                    console.print()
                    render_system("Goodbye!", C_DIM)
                    return False
                _last_ctrl_c = True
                render_system("已取消。（再按一次 Ctrl+C 退出）", C_WARN)
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
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(
    cli_api_key: str | None,
    cli_model: str | None,
    cli_base_url: str | None,
) -> None:
    # Banner
    console.print(Text(BANNER, style=C_BRAND), justify="center")
    console.print(Align.center(Text("Model-agnostic Coding Agent", style=C_DIM)))
    console.print(Align.center(Text("用任意大模型读写和理解代码", style="dim cyan")))
    console.print()

    # Ensure config dir exists before FileHistory tries to open/append to it.
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

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

    # 如果命令行已经提供了完整参数，跳过向导
    if cli_api_key and cli_model:
        config = AgentConfig(
            api_key=cli_api_key,
            model=cli_model,
            base_url=cli_base_url,
            provider_name="CLI",
        )
        await chat_loop(config, pt)
        return

    # 否则进入 配置 → 聊天 → 可重配置 的循环
    config: AgentConfig | None = None
    skip_saved_once = False  # True on /config reconfigure: skip the "use saved?" prompt

    while True:
        # 优先复用已保存的配置，除非用户主动 /config
        saved = None if skip_saved_once else load_config()
        skip_saved_once = False

        if saved is not None:
            masked_key = (saved.api_key[:8] + "..." + saved.api_key[-4:]
                          if len(saved.api_key) > 12 else "****")
            info = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            info.add_column(style=C_DIM)
            info.add_column(style="white")
            info.add_row("Provider", saved.provider_name)
            info.add_row("Model",    saved.model)
            if saved.base_url:
                info.add_row("Base URL", saved.base_url)
            info.add_row("API Key",  masked_key)
            console.print(Panel(
                info, border_style="green",
                title="✓ 检测到已保存的配置", title_align="left",
            ))
            try:
                raw = await pt.prompt_async(
                    "  直接使用? [Y/n]  (n = 重新配置,  f = 忘记并重新配置) › "
                )
            except (KeyboardInterrupt, EOFError):
                console.print()
                render_system("已取消，退出。", C_DIM)
                return
            choice = raw.strip().lower()
            if choice in ("", "y", "yes"):
                config = saved
            elif choice == "f":
                forget_config()
                render_system("✓ 已删除保存的配置。", C_SUCCESS)
                config = None
            else:
                config = None  # fall through to wizard

        if config is None:
            try:
                config = await setup_wizard(pt)
                save_config(config)
                render_system(
                    f"✓ 配置已保存到 {_CONFIG_FILE} (下次启动直接复用)",
                    C_DIM,
                )
            except (KeyboardInterrupt, EOFError):
                console.print()
                render_system("已取消，退出。", C_DIM)
                return

        console.print()
        reconfigure = await chat_loop(config, pt)
        if not reconfigure:
            break
        console.print()
        console.print(Rule("  重新配置", style=C_BRAND))
        config = None
        skip_saved_once = True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Lumen — 代码读写 Agent")
    parser.add_argument("--model",    default=None, help="模型名称，例如 gpt-4o")
    parser.add_argument("--api-key",  default=None, help="API Key（覆盖环境变量）")
    parser.add_argument("--base-url", default=None, help="API Base URL")
    args = parser.parse_args()

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
