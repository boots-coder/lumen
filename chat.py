"""
Lumen — Interactive Code Reading Agent

直接运行，在 UI 内选择模型和输入 API Key：
    python chat.py

也支持命令行直接指定（跳过配置向导）：
    python chat.py --model gpt-4o --api-key sk-...

聊天中的 Slash 命令：
    /help      帮助
    /status    Token 用量和会话信息
    /compact   手动压缩上下文
    /reset     清空对话历史
    /save      保存会话到文件
    /load      从文件加载会话
    /config    重新配置模型和 Key
    /quit      退出
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
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle

from lumen import Agent
from lumen.tools import FileReadTool, GlobTool, GrepTool, BashTool, TreeTool, DefinitionsTool

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
            ModelOption("gpt-4o",         "GPT-4o          128K ctx · 最强综合"),
            ModelOption("gpt-4o-mini",    "GPT-4o Mini     128K ctx · 快速省钱"),
            ModelOption("o3-mini",        "o3-mini         200K ctx · 推理模型"),
            ModelOption("o1",             "o1              200K ctx · 高级推理"),
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

async def setup_wizard(pt: PromptSession) -> AgentConfig:
    """
    交互式配置向导：选 Provider → 选模型 → 输入 Key。
    返回 AgentConfig，失败时抛出 KeyboardInterrupt。
    """
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
        # 自定义：手动输入 base_url 和 model
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
        # 优先使用环境变量中已有的 key
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
    from prompt_toolkit.formatted_text import HTML
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
    if agent.code_reading_mode:
        stats.append("  │  ", style=C_DIM)
        stats.append("⚑ 代码深读", style="bold cyan")

    border = "cyan" if agent.code_reading_mode else C_DIM
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
        ("/mode",         "切换模式：通用 ↔ 代码深读  (/mode code | /mode general)"),
        ("/compact",      "手动压缩上下文"),
        ("/reset",        "清空对话历史"),
        ("/save",         "保存会话到文件"),
        ("/load",         "从文件加载会话"),
        ("/config",       "重新配置模型和 Key"),
        ("/quit",         "退出  (或 Ctrl+C / Ctrl+D)"),
        ("",              ""),
        ("↑ / ↓",        "浏览输入历史"),
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
# 流式输出
# ─────────────────────────────────────────────────────────────────────────────

async def agent_response(agent: Agent, message: str) -> str:
    """
    执行一次完整的 Agent 响应：工具调用过程实时展示，最终答案 Markdown 渲染。
    使用 agent.chat() 以支持完整的工具调用循环。
    """
    tool_log: list[str] = []       # 记录工具调用历史，用于刷新 Live

    def _tool_panel() -> Panel:
        """构建工具调用日志面板。"""
        lines = Text()
        for entry in tool_log[-12:]:          # 最多展示最近 12 条
            lines.append(entry + "\n")
        return Panel(
            lines,
            title=Text("  Lumen  ●  thinking…", style=C_AGENT),
            title_align="left",
            border_style="blue",
            padding=(0, 1),
        )

    def on_tool_call(name: str, arguments: dict) -> None:
        # 把关键参数提炼成一行摘要
        arg_hint = ""
        for key in ("path", "file_path", "pattern", "command", "query"):
            if key in arguments:
                val = str(arguments[key])
                arg_hint = f"  [dim]{val[:60]}{'…' if len(val)>60 else ''}[/]"
                break
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
            transient=True,          # 完成后清除，换成最终答案面板
        ) as live:
            tool_log.append("  [dim]正在思考…[/]")

            async def _chat_with_refresh():
                nonlocal response
                response = await agent.chat(
                    message,
                    on_tool_call=lambda n, a: (on_tool_call(n, a), live.update(_tool_panel())),
                    on_tool_result=lambda n, o, e: (on_tool_result(n, o, e), live.update(_tool_panel())),
                )

            await _chat_with_refresh()
    else:
        # 无工具时走流式输出（体验更好）
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
            async for chunk in agent.stream(message):
                collected.append(chunk)
                md = Markdown("".join(collected), code_theme="monokai", inline_code_theme="monokai")
                live.update(Panel(
                    md, title=panel_title, title_align="left",
                    border_style="blue", padding=(0, 1),
                    width=min(console.width, 120),
                ))
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

def handle_mode(agent: Agent, arg: str) -> None:
    """
    /mode          — 显示当前模式
    /mode code     — 开启代码深读模式
    /mode general  — 切回通用模式
    """
    arg = arg.strip().lower()

    if not arg:
        # 显示当前状态
        mode = "代码深读模式  [Code Reading Mode]" if agent.code_reading_mode else "通用模式  [General]"
        console.print(Panel(
            f"  当前模式: [bold cyan]{mode}[/]\n\n"
            "  [dim]/mode code[/]     → 开启代码深读模式\n"
            "  [dim]/mode general[/]  → 切回通用模式",
            border_style=C_DIM, title="模式", title_align="left",
        ))
        return

    if arg in ("code", "c", "read", "深读"):
        if agent.code_reading_mode:
            render_system("已经是代码深读模式。", C_DIM)
        else:
            agent.enable_code_reading_mode()
            console.print(Panel(
                "[bold cyan]代码深读模式已开启[/]\n\n"
                "  模型现在会:\n"
                "  · 永远先用工具读代码，再回答 (不猜测)\n"
                "  · 每次解释覆盖 WHAT / HOW / WHY / WHERE 四个维度\n"
                "  · 自动追踪调用链并标注 file:line\n"
                "  · 先讲架构全貌，再讲实现细节\n\n"
                "  [dim]通用能力（写代码、回答问题等）仍然完整保留。[/]",
                border_style="cyan", title="✦ Code Reading Mode", title_align="left",
            ))

    elif arg in ("general", "g", "normal", "通用"):
        if not agent.code_reading_mode:
            render_system("已经是通用模式。", C_DIM)
        else:
            agent.disable_code_reading_mode()
            render_system("✓ 已切回通用模式。", C_SUCCESS)

    else:
        render_system(
            f"未知模式 '{arg}'。用 /mode code 或 /mode general。", C_WARN
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


# ─────────────────────────────────────────────────────────────────────────────
# 主聊天循环
# ─────────────────────────────────────────────────────────────────────────────

async def chat_loop(config: AgentConfig, pt: PromptSession) -> bool:
    """
    运行聊天主循环。
    返回 True 表示用户想重新配置（/config），False 表示退出。
    """
    # 创建 Agent（注册全套代码阅读工具）
    try:
        agent = Agent(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            inject_git_state=True,
            inject_memory=True,
            auto_compact=True,
            tools=[
                TreeTool(),         # 项目目录结构树
                DefinitionsTool(),  # 提取文件中所有类/函数定义
                FileReadTool(),     # 按行读取文件内容
                GlobTool(),         # 按模式匹配文件路径
                GrepTool(),         # 搜索文件内容
                BashTool(),         # 执行 shell 命令
            ],
        )
    except Exception as e:
        render_error(f"Agent 创建失败: {e}")
        return True  # 回到配置向导

    # 启动信息
    info = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    info.add_column(style=C_DIM)
    info.add_column(style="white")
    info.add_row("Provider",       config.provider_name)
    info.add_row("Model",          agent.model)
    info.add_row("Context window", f"{agent.context_window:,} tokens")
    info.add_row("Auto-compact",   "✓ 开启")
    info.add_row("Git state",      "✓ 注入")
    info.add_row("Memory files",   "✓ ENGRAM.md 自动发现")
    info.add_row("Tools",          "✓ tree · definitions · read_file · glob · grep · bash")
    console.print(Panel(info, border_style=C_DIM, title="✓ 准备就绪", title_align="left"))
    console.print(Rule(style=C_DIM))
    console.print(
        "  输入消息后按 Enter 发送。"
        "  [magenta]/help[/] 查看命令。"
        "  [dim]Ctrl+D 或 /quit 退出。[/]\n"
    )

    _last_ctrl_c = False

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

        # ── Slash 命令 ────────────────────────────────────────────────────────
        if cmd in ("/quit", "/exit", "/q"):
            render_system("Goodbye!", C_DIM)
            return False

        elif cmd == "/help":
            render_help();  continue

        elif cmd == "/status":
            render_status(agent);  continue

        elif cmd.startswith("/mode"):
            handle_mode(agent, user_input[5:].strip());  continue

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
            continue

        elif cmd == "/config":
            render_system("返回配置向导…", C_DIM)
            return True   # 信号：重新配置

        elif cmd.startswith("/"):
            render_system(f"未知命令: {user_input}  输入 /help 查看帮助", C_WARN)
            continue

        # ── 正常聊天 ──────────────────────────────────────────────────────────
        console.print(render_user_message(user_input))
        console.print()

        try:
            await agent_response(agent, user_input)
            console.print()
        except KeyboardInterrupt:
            if _last_ctrl_c:
                console.print()
                render_system("Goodbye!", C_DIM)
                return False
            _last_ctrl_c = True
            render_system("已取消。（再按一次 Ctrl+C 退出）", C_WARN)
        except Exception as e:
            render_error(str(e))
            if os.environ.get("ENGRAM_DEBUG"):
                traceback.print_exc()


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
    console.print(Align.center(Text("Model-agnostic Deep Code Reading Agent", style=C_DIM)))
    console.print(Align.center(Text("用任意大模型深度阅读和理解代码", style="dim cyan")))
    console.print()

    pt = PromptSession(
        history=InMemoryHistory(),
        style=PTStyle.from_dict({"prompt": "ansicyan bold"}),
    )

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

    while True:
        try:
            # 尝试从环境变量中自动填充（仍然走向导让用户确认）
            config = await setup_wizard(pt)
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Lumen — 代码深度阅读 Agent")
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
