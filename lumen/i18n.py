"""
Lightweight runtime i18n for the Lumen chat UI.

Design goals:
  · No external deps (no gettext, no babel) — ships with Lumen.
  · A single module-level `t(key, **kw)` call from UI code.
  · Language switch is runtime-live: `/lang en` / `/lang zh` flips every
    subsequent rendered string without re-creating the agent or restarting
    the REPL.
  · Persist the user's choice in ~/.lumen/config.json so the next launch
    picks up where they left off.
  · LLM-facing prompts are kept in English everywhere — i18n only touches
    strings the user reads in the terminal.
  · Missing keys fall back: zh → en → raw key. Never raises.

Key convention:
  dotted namespace, e.g. `"help.table.title"`, `"wizard.step.provider"`.
  Values may contain `{named}` placeholders resolved by `str.format`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "zh")
DEFAULT_LANGUAGE: str = "en"

_current: str = DEFAULT_LANGUAGE


# ─────────────────────────────────────────────────────────────────────────────
# Translation tables
# ─────────────────────────────────────────────────────────────────────────────
# Keep keys sorted by namespace. Add new keys to BOTH tables.

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # ── banner / tagline ────────────────────────────────────────────────
        "banner.tagline": "Model-agnostic Coding Agent",
        "banner.subtitle": "Read and understand code with any LLM",

        # ── generic words ───────────────────────────────────────────────────
        "common.cancelled": "Cancelled",
        "common.goodbye": "Goodbye!",
        "common.enabled": "enabled",
        "common.disabled": "off",
        "common.on": "on",
        "common.none": "none",
        "common.yes": "yes",
        "common.no": "no",

        # ── argparse / main ─────────────────────────────────────────────────
        "cli.description": "Lumen — code reading/writing Agent",
        "cli.arg.model": "Model name, e.g. gpt-4o",
        "cli.arg.api_key": "API Key (overrides env var)",
        "cli.arg.base_url": "API Base URL",
        "cli.arg.lang": "UI language (en | zh)",

        # ── wizard ──────────────────────────────────────────────────────────
        "wizard.title": "  Configure model",
        "wizard.reconfigure": "  Reconfigure",
        "wizard.pick_provider": "Pick a Provider",
        "wizard.pick_model": "Pick a model",
        "wizard.enter_number": "  Enter # \u203a ",
        "wizard.range_hint": "  [yellow]Please enter a number between 1 and {n}[/]",
        "wizard.must_be_number": "  [yellow]Please enter a number[/]",
        "wizard.selected": "  [green]\u2713 Selected:[/]{value}",
        "wizard.key_env_detected": "  Env var {name} detected ({masked})\n  Use it? [Y/n] \u203a ",
        "wizard.key_env_used": "  [green]\u2713 Using key from env var[/]",
        "wizard.key_hint_prefix": "  [dim]Format example: {hint}[/]",
        "wizard.key_prompt": "  API Key \u203a ",
        "wizard.key_empty": "  [yellow]Key cannot be empty, please retry[/]",
        "wizard.local_no_key": "  [dim]Local model does not need an API key; using placeholder[/]\n",
        "wizard.custom_base_url": "  Base URL (e.g. http://localhost:11434/v1) \u203a ",
        "wizard.custom_model": "  Model name \u203a ",
        "wizard.custom_empty_model": "model cannot be empty",
        "wizard.summary_title": "\u2713 Configuration confirmed",
        "wizard.saved_to": "\u2713 Config saved to {path} (will be reused next launch)",
        "wizard.save_failed": "(could not save config: {err})",
        "wizard.cancel_exit": "Cancelled, exiting.",
        "wizard.saved_detected": "\u2713 Saved configuration detected",
        "wizard.use_saved_prompt": "  Use it? [Y/n]  (n = reconfigure, f = forget & reconfigure) \u203a ",
        "wizard.forgot": "\u2713 Saved configuration deleted.",

        # ── provider labels (only the ones that need translation) ───────────
        "provider.getgoapi.name": "GetGoAPI (multi-model proxy)",
        "provider.ollama.name": "Ollama (local models)",
        "provider.custom.name": "Custom / Other",
        "provider.ollama.key_hint": "No key needed",
        "provider.custom.key_hint": "Your API Key",
        "provider.model.latest_strongest": "latest strongest all-round",
        "provider.model.fast_cheap": "fast & cheap",
        "provider.model.ultra_cheap": "ultra-fast, ultra-cheap",
        "provider.model.allround": "all-round capability",
        "provider.model.reasoning": "reasoning model",
        "provider.model.sonnet_strongest": "Claude Sonnet 4.6  strongest all-round",
        "provider.model.haiku_fastcheap": "Claude Haiku 4.5   fast & cheap",
        "provider.model.deepseek_chat": "DeepSeek Chat   general chat",
        "provider.model.deepseek_reasoner": "DeepSeek Reasoner reasoning model",

        # ── ready / startup info panel ──────────────────────────────────────
        "startup.ready": "\u2713 Ready",
        "startup.auto_compact": "\u2713 on",
        "startup.git_state": "\u2713 injected",
        "startup.memory_files": "\u2713 ENGRAM.md auto-discovered",
        "startup.tools_read": "\u2713 tree \u00b7 definitions \u00b7 read_file \u00b7 glob \u00b7 grep",
        "startup.tools_write": "[yellow]\u270e[/] write_file \u00b7 edit_file \u00b7 bash",
        "startup.permissions": "\u2713 read silent \u00b7 write needs confirm \u00b7 dangerous cmd denied",
        "startup.retry": "\u2713 exp. backoff \u00b7 429/529 retry \u00b7 max_tokens auto-shrink",
        "startup.file_cache": "\u2713 LRU 100 files/25MB",
        "startup.notifier_on": "\u2713 {channel}  (>{sec}s auto-ring)",
        "startup.notifier_off": "off  (LUMEN_NOTIFY=0)",
        "startup.mcp_connected": "\u2713 MCP connected: {names}",
        "startup.mcp_failed": "  \u26a0 MCP {name} connect failed: {err}",
        "startup.mcp_load_failed": "  \u26a0 MCP load failed: {err}",
        "startup.hint": "  Press Enter to send. [magenta]/help[/] lists commands. [dim]Ctrl+D or /quit exits.[/]\n",
        "startup.auto_save": "\u2713 {path}",

        # ── column headings used in startup info ────────────────────────────
        "col.provider": "Provider",
        "col.model": "Model",
        "col.base_url": "Base URL",
        "col.api_key": "API Key",
        "col.session_id": "Session ID",
        "col.context_window": "Context window",
        "col.auto_compact": "Auto-compact",
        "col.auto_save": "Auto-save",
        "col.git_state": "Git state",
        "col.memory_files": "Memory files",
        "col.tools_read": "Tools (read)",
        "col.tools_write": "Tools (write)",
        "col.permissions": "Permissions",
        "col.retry": "Retry/Fallback",
        "col.file_cache": "File cache",
        "col.desktop_notify": "Desktop notify",
        "col.language": "Language",

        # ── help ────────────────────────────────────────────────────────────
        "help.title": "Commands",
        "help.row.history": "Scroll input history (persisted at ~/.lumen/history)",
        "help.row.shift_enter": "Newline (requires terminal support; iTerm2 users: use Alt+Enter)",
        "help.row.alt_enter": "Newline (cross-terminal fallback, same as Ctrl+J)",
        "help.row.at_path": "Auto-complete project file paths",
        "help.row.shift_tab": "Cycle general \u2192 plan \u2192 review",
        "help.row.ctrl_c": "Cancel current input / press again to quit",

        # ── status ─────────────────────────────────────────────────────────
        "status.title": "Session Status",
        "status.model": "Model",
        "status.context_window": "Context window",
        "status.tokens_used": "Tokens used",
        "status.messages": "Messages",
        "status.compactions": "Compactions",
        "status.session_start": "Session start",

        # ── token bar labels ────────────────────────────────────────────────
        "tokenbar.model": "  model: ",
        "tokenbar.msgs": "  \u2502  msgs: ",
        "tokenbar.compactions": "  \u2502  compactions: ",
        "tokenbar.review_idle": "review \u00b7 idle",
        "tokenbar.review_design": "review \u00b7 design",
        "tokenbar.review_pipeline": "review \u00b7 pipeline",
        "tokenbar.review_impl": "review \u00b7 impl({n})",
        "tokenbar.review_complete": "review \u00b7 complete",
        "tokenbar.review_default": "review",
        "tokenbar.plan_mode": "\U0001f4cb Plan Mode",

        # ── mode switching ──────────────────────────────────────────────────
        "mode.review_label": "Review Mode",
        "mode.plan_label": "Plan Mode",
        "mode.general_label": "General",
        "mode.current": "  Current mode: [bold cyan]{label}[/]",
        "mode.hint_review": "  [dim]/mode review[/]   \u2192 enable Review Mode (3-phase gate)",
        "mode.hint_plan": "  [dim]/mode plan[/]     \u2192 enable Plan Mode (read-only research + plan)",
        "mode.hint_general": "  [dim]/mode general[/]  \u2192 back to general mode",
        "mode.hint_shift_tab": "  [dim]Shift+Tab[/]       \u2192 cycle general / plan / review",
        "mode.panel_title": "Mode",
        "mode.already_review": "Already in Review Mode.",
        "mode.already_plan": "Already in Plan Mode.",
        "mode.already_general": "Already in General Mode.",
        "mode.back_to_general": "\u2713 Back to General Mode.",
        "mode.unknown": "Unknown mode '{arg}'. Use /mode review | /mode plan | /mode general.",
        "mode.review_panel_title": "\u2691 Review Mode",
        "mode.review_panel_body": (
            "[bold yellow]Review Mode enabled \u2014 real phase gating[/]\n\n"
            "  When writing code the model proceeds in 3 steps and will "
            "[bold]call the request_approval tool[/] to pause and wait for you:\n"
            "  \u00b7 Phase 1  [cyan]Design[/]    \u2014 show function/class signatures and responsibilities\n"
            "  \u00b7 Phase 2  [blue]Data flow[/] \u2014 arrow-based pipeline diagram\n"
            "  \u00b7 Phase 3  [yellow]Impl[/]    \u2014 one function at a time, explaining data transitions\n\n"
            "  When the approval panel pops up you can:\n"
            "  \u00b7 [bold]Y / bare Enter[/] \u2192 approve, go to next step\n"
            "  \u00b7 [bold]skip / write all[/] \u2192 approve and skip remaining approvals\n"
            "  \u00b7 [bold]N[/] then type feedback \u2192 reject and revise\n"
            "  \u00b7 Type any free text \u2192 treated as feedback and rejected\n\n"
            "  [dim]The token bar shows the current phase. Non-coding tasks are unaffected.[/]"
        ),
        "mode.plan_panel_title": "\U0001f4cb Plan Mode",
        "mode.plan_panel_body": (
            "[bold blue]Plan Mode enabled \u2014 read-only research + structured plan[/]\n\n"
            "  This turn the model will only: [cyan]read_file \u00b7 glob \u00b7 tree \u00b7 grep[/]; no files are written.\n"
            "  The reply ends with a structured plan (Goal / Files / Steps / Risks).\n\n"
            "  \u00b7 [bold]/plan approve[/]  \u2192 approve plan, auto-switch to acceptEdits and start executing\n"
            "  \u00b7 [bold]/plan cancel[/]   \u2192 cancel, return to general mode\n"
            "  \u00b7 [bold]Shift+Tab[/]       \u2192 cycle modes"
        ),

        # ── /plan command ───────────────────────────────────────────────────
        "plan.on_title": "Plan Mode",
        "plan.active_body": (
            "[bold blue]\U0001f4cb Plan Mode is active[/]\n\n"
            "  \u00b7 [bold]/plan approve[/]  \u2192 approve plan, switch to acceptEdits and execute\n"
            "  \u00b7 [bold]/plan cancel[/]   \u2192 cancel, back to general mode"
        ),
        "plan.inactive_body": (
            "Plan Mode is off.\n\n"
            "  \u00b7 [bold]/plan on[/]       \u2192 enable (read-only research + plan)\n"
            "  \u00b7 or [bold]/mode plan[/]\n"
            "  \u00b7 or Shift+Tab to cycle modes"
        ),
        "plan.off_not_active": "Plan Mode is not active.",
        "plan.cancelled": "\u2713 Plan Mode cancelled; back to general mode.",
        "plan.approve_not_active": "Plan Mode is not active \u2014 /plan on or /mode plan first.",
        "plan.approved": "\u2713 Plan approved \u2014 switched to [bold green]acceptEdits[/] and starting execution.",
        "plan.unknown_sub": "Unknown subcommand '{sub}'. Options: on / off / approve / cancel.",

        # ── compact ─────────────────────────────────────────────────────────
        "compact.running": "Compacting context\u2026",
        "compact.title": "Compaction result",
        "compact.body": (
            "[green]\u2713 Compaction done[/]\n\n"
            "  Messages: [white]{mb}[/] \u2192 [white]{ma}[/]\n"
            "  Tokens  : [white]{tb:,}[/] \u2192 [white]{ta:,}[/]\n"
            "  Kept last [white]{kept}[/] raw messages"
        ),
        "compact.failed": "Compaction failed: {err}",

        # ── save / load / resume ───────────────────────────────────────────
        "save.prompt": "  Save to [{default}] \u203a ",
        "save.done": "\u2713 Saved \u2192 {path}",
        "save.transcript_hint": "  (session is also auto-saved at {path})",
        "save.failed": "Save failed: {err}",
        "load.prompt": "  File path to load \u203a ",
        "load.missing": "File not found: {path!r}",
        "load.done": "\u2713 Loaded {n} messages from {path}",
        "load.failed": "Load failed: {err}",
        "resume.none": "No auto-saved sessions found.",
        "resume.table_title": "Recent auto-saved sessions",
        "resume.pick_prompt": "  Enter # to resume (or press Enter to cancel) \u203a ",
        "resume.range_hint": "Please enter a number between 1 and {n}",
        "resume.empty_file": "Session file is empty.",
        "resume.done": "\u2713 Resumed session {sid}\u2026 ({n} messages)",
        "resume.failed": "Resume failed: {err}",
        "resume.col.session_id": "Session ID",
        "resume.col.messages": "Messages",
        "resume.col.created": "Created",
        "resume.col.last_prompt": "Last prompt",

        # ── reset / config / forget ────────────────────────────────────────
        "reset.done": "\u2713 Conversation history cleared.",
        "config.back_to_wizard": "Returning to configuration wizard\u2026",
        "forget.done": "\u2713 Deleted saved configuration ({path})",
        "forget.nothing": "No saved configuration to delete.",

        # ── permissions command ────────────────────────────────────────────
        "perm.current": "  Current: [bold cyan]{name}[/]\n",
        "perm.label.default": ("default", "Normal: write ops ask each time, dangerous cmds denied"),
        "perm.label.accept_edits": ("acceptEdits", "write/edit auto-approved, bash still checked"),
        "perm.label.plan": ("plan", "Plan: read-only research, write/edit/bash all denied"),
        "perm.label.bypass": ("bypass", "Bypass: all tools auto-approved \u26a0 dangerous"),
        "perm.panel_title": "Permission Mode",
        "perm.unknown": "Unknown permission mode '{arg}'. /perm shows options.",
        "perm.already": "Already in {name} mode.",
        "perm.bypass_confirm": "  \u26a0 bypass mode skips all permission checks \u2014 confirm? [y/N] \u203a ",
        "perm.switched": "\u2713 Permission mode set to [bold]{name}[/] \u2014 {desc}",

        # ── permission prompt (write/risky tool) ──────────────────────────
        "permission.panel_title": "\U0001f512 Confirmation required",
        "permission.panel_body": "  Tool: [bold yellow]{tool}[/]\n{hint}",
        "permission.prompt": "  Allow? [Y/n] \u203a ",

        # ── diff ───────────────────────────────────────────────────────────
        "diff.no_tracker": "File tracker not ready.",
        "diff.no_changes": "No files were written or edited this session.",
        "diff.no_match": "No modified files match '{target}'.",
        "diff.col.file": "File",
        "diff.col.plus": "+",
        "diff.col.minus": "-",
        "diff.col.writes": "Writes",
        "diff.col.status": "Status",
        "diff.status_new": "new",
        "diff.status_edited": "edited",
        "diff.panel_title": "File changes \u00b7 {n} file(s) \u00b7 +{added}/-{removed}",

        # ── tasks ──────────────────────────────────────────────────────────
        "tasks.cleared": "\u2713 All background tasks cleared.",
        "tasks.stopped": "\u2713 Task {id} stopped.",
        "tasks.stop_missing": "Task {id!r} not found or already ended.",
        "tasks.not_found": "No task with id = {id!r}",
        "tasks.empty_hint": "No background tasks. The agent can call sub_agent(run_in_background=True) to start one.",
        "tasks.col.state": "Status",
        "tasks.col.elapsed": "Elapsed",
        "tasks.col.description": "Description",
        "tasks.table_title": "Background tasks \u00b7 {n} total (running {r})",
        "tasks.footer_hint": "Tip: /tasks <id> to view \u00b7 /tasks stop <id> to cancel \u00b7 /tasks clear to wipe",

        # ── dream ──────────────────────────────────────────────────────────
        "dream.disabled_hint": "Auto-Dream is off. It starts by default; use agent.enable_auto_dream() to enable.",
        "dream.off": "\u2713 Auto-Dream disabled.",
        "dream.forcing": "Forcing a dream run\u2026",
        "dream.forced_done": "\u2713 dream done \u2014 extractions={e} consolidations={c}",
        "dream.consolidating": "Consolidating\u2026",
        "dream.consolidate_empty": "Not enough entries to consolidate.",
        "dream.consolidate_title": "Latest consolidation summary",
        "dream.stats_title": "Auto-Dream",
        "dream.stats_state_dreaming": "[yellow]\u25cf dreaming[/]",
        "dream.stats_state_idle": "[green]\u25cf idle[/]",
        "dream.stats_body": (
            "  State       : {state}\n"
            "  Extractions : {e}\n"
            "  Consolidations : {c}\n"
            "  Last run    : {last}\n"
            "  Last error  : {err}\n"
            "  Memory entries : {mem}\n\n"
            "  Interval (turns) : {interval}\n"
            "  Consolidate every: {cons} extractions\n\n"
            "  [dim]/dream force[/]        force one run\n"
            "  [dim]/dream consolidate[/]  consolidation only\n"
            "  [dim]/dream off[/]          disable"
        ),

        # ── mcp ────────────────────────────────────────────────────────────
        "mcp.no_server": "No MCP server {name!r}",
        "mcp.reconnect_ok": "\u2713 {name} reconnected \u2014 {n} tools",
        "mcp.reconnect_failed": "\u2717 {name} reconnect failed: {err}",
        "mcp.disconnected": "\u2713 {name} disconnected.",
        "mcp.not_connected": "{name!r} is not connected.",
        "mcp.empty_body": (
            "No MCP servers are currently connected.\n\n"
            "To enable, edit [cyan]{path}[/] using this shape:\n\n"
            "[dim]{{\n"
            '  "mcpServers": {{\n'
            '    "filesystem": {{\n'
            '      "command": "npx",\n'
            '      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]\n'
            "    }}\n"
            "  }}\n"
            "}}[/]\n\n"
            "Then run /mcp reconnect <name> or restart lumen."
        ),
        "mcp.col.status": "Status",
        "mcp.col.tools": "Tools",
        "mcp.col.detail": "Error / command",
        "mcp.table_title": "MCP servers \u00b7 {n} total",
        "mcp.footer_hint": "Tip: /mcp reconnect <name> to reconnect \u00b7 /mcp disconnect <name> to disconnect",

        # ── vim ────────────────────────────────────────────────────────────
        "vim.off": "\u2713 Vim mode off (back to Emacs/default)",
        "vim.on": "\u2713 Vim mode on  \u00b7  Esc\u2192Normal  \u00b7  i\u2192Insert  \u00b7  :\u2192colon command",

        # ── /lang ─────────────────────────────────────────────────────────
        "lang.current": "Current language: [bold cyan]{name}[/]",
        "lang.switched": "\u2713 Language switched to [bold cyan]{name}[/]",
        "lang.already": "Already in [bold cyan]{name}[/].",
        "lang.unknown": "Unknown language '{arg}'. Available: en | zh.",
        "lang.available": "Available: [bold]/lang en[/] \u00b7 [bold]/lang zh[/]",
        "lang.name.en": "English",
        "lang.name.zh": "\u4e2d\u6587",

        # ── slash command catalog (/help descriptions) ─────────────────────
        "cmd.help": "Show help",
        "cmd.status": "Token usage and session info",
        "cmd.mode": "Switch mode (general | plan | review)",
        "cmd.plan": "Plan Mode control (on/off/approve/cancel)",
        "cmd.diff": "Show files the agent has modified this session (optional path filter)",
        "cmd.tasks": "Background tasks (list / view id / stop id / clear)",
        "cmd.mcp": "MCP servers (list / reconnect / disconnect)",
        "cmd.dream": "Auto-Dream memory consolidation (force/consolidate/off)",
        "cmd.perm": "Permission mode (default/acceptEdits/plan/bypass)",
        "cmd.compact": "Manually compact context",
        "cmd.reset": "Clear conversation history",
        "cmd.save": "Save session to file",
        "cmd.load": "Load session from file",
        "cmd.resume": "Resume the most recent auto-saved session",
        "cmd.config": "Reconfigure model and key",
        "cmd.forget": "Delete locally saved API key",
        "cmd.vim": "Toggle Vim mode",
        "cmd.lang": "Switch UI language (en | zh)",
        "cmd.quit": "Quit",

        # ── command subarg menu labels ─────────────────────────────────────
        "cmd.sub.mode.general": "General Mode",
        "cmd.sub.mode.plan": "Plan Mode (read-only research)",
        "cmd.sub.mode.review": "Review Mode (phase gating)",
        "cmd.sub.plan.on": "Enable plan mode",
        "cmd.sub.plan.approve": "Approve plan and start executing",
        "cmd.sub.plan.cancel": "Cancel plan mode",
        "cmd.sub.plan.off": "Same as cancel",
        "cmd.sub.tasks.clear": "Clear all tasks",
        "cmd.sub.tasks.stop": "/tasks stop <id> to cancel a task",
        "cmd.sub.mcp.reconnect": "/mcp reconnect <name>",
        "cmd.sub.mcp.disconnect": "/mcp disconnect <name>",
        "cmd.sub.dream.force": "Force one run",
        "cmd.sub.dream.consolidate": "Consolidation only",
        "cmd.sub.dream.off": "Disable Auto-Dream",
        "cmd.sub.perm.default": "Write ops ask each time",
        "cmd.sub.perm.accept_edits": "write/edit auto-approved",
        "cmd.sub.perm.plan": "Read-only research",
        "cmd.sub.perm.bypass": "\u26a0 allow everything",
        "cmd.sub.lang.en": "Switch to English",
        "cmd.sub.lang.zh": "Switch to Chinese",

        # ── bottom toolbar ─────────────────────────────────────────────────
        "toolbar.review_idle": "idle",
        "toolbar.review_design": "design",
        "toolbar.review_pipeline": "pipeline",
        "toolbar.review_impl": "impl\u00d7{n}",
        "toolbar.review_complete": "complete",
        "toolbar.mode_review": " \u2691 review \u00b7 {phase} ",
        "toolbar.mode_plan": " \U0001f4cb plan ",
        "toolbar.mode_general": " general ",
        "toolbar.perm_accept_edits": " \u270e accept-edits \u2502",
        "toolbar.perm_plan": " \U0001f4cb plan-only \u2502",
        "toolbar.perm_bypass": " \u26a0 BYPASS \u2502",
        "toolbar.line": (
            " mode:{mode}\u2502{perm} \u21e7\u21e5 mode \u2502 / cmds \u2502 @ files \u2502 "
            "\u21e7\u23ce / \u2325\u23ce newline \u2502 {model} "
        ),

        # ── generic messages ───────────────────────────────────────────────
        "msg.unknown_command": "Unknown command: {cmd}  type /help for help",
        "msg.cmd_failed": "{cmd} failed: {err}",
        "msg.agent_failed": "Agent creation failed: {err}",
        "msg.cancel_reply": "Cancelled. (press Ctrl+C again to quit)",
        "msg.user_cmd_load_failed": "  \u26a0 user command {file} failed to load: {err}",
        "msg.user_cmd_loaded": "\u2713 Loaded user commands: {names}",
        "msg.reply_done": "Reply done \u00b7 {preview}",
        "msg.thinking": "  [dim]Thinking\u2026[/]",
        "msg.lumen_thinking": "  Lumen  \u25cf  thinking\u2026",
        "msg.lumen_streaming": "  Lumen  \u25cf  streaming\u2026",
        "msg.lumen": "  Lumen",
        "msg.you": "  You",

        # ── approval handler ───────────────────────────────────────────────
        "approval.prompt": "Enter=approve  n=reject  skip=approve all  (or type feedback) \u203a ",
        "approval.feedback_prompt": "Feedback \u203a ",
    },

    "zh": {
        # ── banner / tagline ────────────────────────────────────────────────
        "banner.tagline": "Model-agnostic Coding Agent",
        "banner.subtitle": "\u7528\u4efb\u610f\u5927\u6a21\u578b\u8bfb\u5199\u548c\u7406\u89e3\u4ee3\u7801",

        # ── generic words ───────────────────────────────────────────────────
        "common.cancelled": "\u5df2\u53d6\u6d88",
        "common.goodbye": "Goodbye!",
        "common.enabled": "\u5f00\u542f",
        "common.disabled": "\u5173\u95ed",
        "common.on": "\u5f00",
        "common.none": "\u65e0",
        "common.yes": "\u662f",
        "common.no": "\u5426",

        # ── argparse / main ─────────────────────────────────────────────────
        "cli.description": "Lumen \u2014 \u4ee3\u7801\u8bfb\u5199 Agent",
        "cli.arg.model": "\u6a21\u578b\u540d\u79f0\uff0c\u4f8b\u5982 gpt-4o",
        "cli.arg.api_key": "API Key\uff08\u8986\u76d6\u73af\u5883\u53d8\u91cf\uff09",
        "cli.arg.base_url": "API Base URL",
        "cli.arg.lang": "\u754c\u9762\u8bed\u8a00 (en | zh)",

        # ── wizard ──────────────────────────────────────────────────────────
        "wizard.title": "  \u914d\u7f6e\u6a21\u578b",
        "wizard.reconfigure": "  \u91cd\u65b0\u914d\u7f6e",
        "wizard.pick_provider": "\u9009\u62e9 Provider",
        "wizard.pick_model": "\u9009\u62e9\u6a21\u578b",
        "wizard.enter_number": "  \u8f93\u5165\u7f16\u53f7 \u203a ",
        "wizard.range_hint": "  [yellow]\u8bf7\u8f93\u5165 1\u2013{n} \u4e4b\u95f4\u7684\u6570\u5b57[/]",
        "wizard.must_be_number": "  [yellow]\u8bf7\u8f93\u5165\u6570\u5b57[/]",
        "wizard.selected": "  [green]\u2713 \u5df2\u9009\uff1a[/]{value}",
        "wizard.key_env_detected": "  \u68c0\u6d4b\u5230\u73af\u5883\u53d8\u91cf {name} ({masked})\n  \u76f4\u63a5\u4f7f\u7528? [Y/n] \u203a ",
        "wizard.key_env_used": "  [green]\u2713 \u4f7f\u7528\u73af\u5883\u53d8\u91cf\u4e2d\u7684 Key[/]",
        "wizard.key_hint_prefix": "  [dim]\u683c\u5f0f\u53c2\u8003\uff1a{hint}[/]",
        "wizard.key_prompt": "  API Key \u203a ",
        "wizard.key_empty": "  [yellow]Key \u4e0d\u80fd\u4e3a\u7a7a\uff0c\u8bf7\u91cd\u65b0\u8f93\u5165[/]",
        "wizard.local_no_key": "  [dim]\u672c\u5730\u6a21\u578b\u65e0\u9700 API Key\uff0c\u4f7f\u7528\u9ed8\u8ba4\u5360\u4f4d\u7b26[/]\n",
        "wizard.custom_base_url": "  Base URL (\u5982 http://localhost:11434/v1) \u203a ",
        "wizard.custom_model": "  Model \u540d\u79f0 \u203a ",
        "wizard.custom_empty_model": "model \u4e0d\u80fd\u4e3a\u7a7a",
        "wizard.summary_title": "\u2713 \u914d\u7f6e\u786e\u8ba4",
        "wizard.saved_to": "\u2713 \u914d\u7f6e\u5df2\u4fdd\u5b58\u5230 {path} (\u4e0b\u6b21\u542f\u52a8\u76f4\u63a5\u590d\u7528)",
        "wizard.save_failed": "(\u65e0\u6cd5\u4fdd\u5b58\u914d\u7f6e: {err})",
        "wizard.cancel_exit": "\u5df2\u53d6\u6d88\uff0c\u9000\u51fa\u3002",
        "wizard.saved_detected": "\u2713 \u68c0\u6d4b\u5230\u5df2\u4fdd\u5b58\u7684\u914d\u7f6e",
        "wizard.use_saved_prompt": "  \u76f4\u63a5\u4f7f\u7528? [Y/n]  (n = \u91cd\u65b0\u914d\u7f6e,  f = \u5fd8\u8bb0\u5e76\u91cd\u65b0\u914d\u7f6e) \u203a ",
        "wizard.forgot": "\u2713 \u5df2\u5220\u9664\u4fdd\u5b58\u7684\u914d\u7f6e\u3002",

        # ── provider labels ────────────────────────────────────────────────
        "provider.getgoapi.name": "GetGoAPI (\u591a\u6a21\u578b\u4ee3\u7406)",
        "provider.ollama.name": "Ollama (\u672c\u5730\u6a21\u578b)",
        "provider.custom.name": "\u81ea\u5b9a\u4e49 / \u5176\u4ed6",
        "provider.ollama.key_hint": "\u65e0\u9700 Key",
        "provider.custom.key_hint": "\u4f60\u7684 API Key",
        "provider.model.latest_strongest": "\u6700\u65b0\u6700\u5f3a\u7efc\u5408",
        "provider.model.fast_cheap": "\u5feb\u901f\u7701\u94b1",
        "provider.model.ultra_cheap": "\u6781\u901f\u8d85\u7701",
        "provider.model.allround": "\u7efc\u5408\u80fd\u529b",
        "provider.model.reasoning": "\u63a8\u7406\u6a21\u578b",
        "provider.model.sonnet_strongest": "Claude Sonnet 4.6  \u6700\u5f3a\u7efc\u5408",
        "provider.model.haiku_fastcheap": "Claude Haiku 4.5   \u5feb\u901f\u7701\u94b1",
        "provider.model.deepseek_chat": "DeepSeek Chat   \u901a\u7528\u5bf9\u8bdd",
        "provider.model.deepseek_reasoner": "DeepSeek Reasoner \u63a8\u7406\u6a21\u578b",

        # ── ready / startup info panel ──────────────────────────────────────
        "startup.ready": "\u2713 \u51c6\u5907\u5c31\u7eea",
        "startup.auto_compact": "\u2713 \u5f00\u542f",
        "startup.git_state": "\u2713 \u6ce8\u5165",
        "startup.memory_files": "\u2713 ENGRAM.md \u81ea\u52a8\u53d1\u73b0",
        "startup.tools_read": "\u2713 tree \u00b7 definitions \u00b7 read_file \u00b7 glob \u00b7 grep",
        "startup.tools_write": "[yellow]\u270e[/] write_file \u00b7 edit_file \u00b7 bash",
        "startup.permissions": "\u2713 \u8bfb\u64cd\u4f5c\u9759\u9ed8\u653e\u884c \u00b7 \u5199\u64cd\u4f5c\u9700\u786e\u8ba4 \u00b7 \u5371\u9669\u547d\u4ee4\u62d2\u7edd",
        "startup.retry": "\u2713 \u6307\u6570\u9000\u907f \u00b7 429/529\u91cd\u8bd5 \u00b7 max_tokens\u81ea\u52a8\u7f29\u51cf",
        "startup.file_cache": "\u2713 LRU 100\u6587\u4ef6/25MB",
        "startup.notifier_on": "\u2713 {channel}  (>{sec}s \u81ea\u52a8\u54cd)",
        "startup.notifier_off": "\u5173\u95ed  (LUMEN_NOTIFY=0)",
        "startup.mcp_connected": "\u2713 MCP \u5df2\u8fde\u63a5: {names}",
        "startup.mcp_failed": "  \u26a0 MCP {name} \u8fde\u63a5\u5931\u8d25: {err}",
        "startup.mcp_load_failed": "  \u26a0 MCP \u52a0\u8f7d\u5931\u8d25: {err}",
        "startup.hint": "  \u8f93\u5165\u6d88\u606f\u540e\u6309 Enter \u53d1\u9001\u3002  [magenta]/help[/] \u67e5\u770b\u547d\u4ee4\u3002  [dim]Ctrl+D \u6216 /quit \u9000\u51fa\u3002[/]\n",
        "startup.auto_save": "\u2713 {path}",

        # ── column headings used in startup info ────────────────────────────
        "col.provider": "Provider",
        "col.model": "Model",
        "col.base_url": "Base URL",
        "col.api_key": "API Key",
        "col.session_id": "Session ID",
        "col.context_window": "Context window",
        "col.auto_compact": "Auto-compact",
        "col.auto_save": "Auto-save",
        "col.git_state": "Git state",
        "col.memory_files": "Memory files",
        "col.tools_read": "Tools (\u8bfb)",
        "col.tools_write": "Tools (\u5199)",
        "col.permissions": "\u6743\u9650\u7cfb\u7edf",
        "col.retry": "\u91cd\u8bd5/\u964d\u7ea7",
        "col.file_cache": "\u6587\u4ef6\u7f13\u5b58",
        "col.desktop_notify": "\u684c\u9762\u901a\u77e5",
        "col.language": "\u8bed\u8a00",

        # ── help ────────────────────────────────────────────────────────────
        "help.title": "\u547d\u4ee4\u5217\u8868",
        "help.row.history": "\u6d4f\u89c8\u8f93\u5165\u5386\u53f2\uff08\u6301\u4e45\u5316\u5728 ~/.lumen/history\uff09",
        "help.row.shift_enter": "\u6362\u884c\uff08\u9700\u7ec8\u7aef\u652f\u6301\uff1biTerm2 \u7528\u6237\u53ef\u6539\u7528 Alt+Enter\uff09",
        "help.row.alt_enter": "\u6362\u884c\uff08\u8de8\u7ec8\u7aef\u515c\u5e95\uff0cCtrl+J \u540c\u6548\uff09",
        "help.row.at_path": "\u81ea\u52a8\u8865\u5168\u9879\u76ee\u5185\u6587\u4ef6\u8def\u5f84",
        "help.row.shift_tab": "\u5faa\u73af\u5207\u6362 general \u2192 plan \u2192 review",
        "help.row.ctrl_c": "\u53d6\u6d88\u5f53\u524d\u8f93\u5165 / \u518d\u6309\u4e00\u6b21\u9000\u51fa",

        # ── status ─────────────────────────────────────────────────────────
        "status.title": "Session Status",
        "status.model": "Model",
        "status.context_window": "Context window",
        "status.tokens_used": "Tokens used",
        "status.messages": "Messages",
        "status.compactions": "Compactions",
        "status.session_start": "Session start",

        # ── token bar labels ────────────────────────────────────────────────
        "tokenbar.model": "  model: ",
        "tokenbar.msgs": "  \u2502  msgs: ",
        "tokenbar.compactions": "  \u2502  compactions: ",
        "tokenbar.review_idle": "\u5ba1\u9605\u00b7\u5f85\u547d",
        "tokenbar.review_design": "\u5ba1\u9605\u00b7\u8bbe\u8ba1",
        "tokenbar.review_pipeline": "\u5ba1\u9605\u00b7\u6570\u636e\u6d41",
        "tokenbar.review_impl": "\u5ba1\u9605\u00b7\u5b9e\u73b0({n})",
        "tokenbar.review_complete": "\u5ba1\u9605\u00b7\u5b8c\u6210",
        "tokenbar.review_default": "\u5ba1\u9605",
        "tokenbar.plan_mode": "\U0001f4cb \u65b9\u6848\u6a21\u5f0f",

        # ── mode switching ──────────────────────────────────────────────────
        "mode.review_label": "\u5ba1\u9605\u6a21\u5f0f  [Review Mode]",
        "mode.plan_label": "\u65b9\u6848\u6a21\u5f0f  [Plan Mode]",
        "mode.general_label": "\u901a\u7528\u6a21\u5f0f  [General]",
        "mode.current": "  \u5f53\u524d\u6a21\u5f0f: [bold cyan]{label}[/]",
        "mode.hint_review": "  [dim]/mode review[/]   \u2192 \u5f00\u542f\u5ba1\u9605\u6a21\u5f0f\uff083 \u9636\u6bb5\u95e8\u63a7\uff09",
        "mode.hint_plan": "  [dim]/mode plan[/]     \u2192 \u5f00\u542f\u65b9\u6848\u6a21\u5f0f\uff08\u53ea\u8bfb\u8c03\u7814 + \u7ed3\u6784\u5316\u65b9\u6848\uff09",
        "mode.hint_general": "  [dim]/mode general[/]  \u2192 \u5207\u56de\u901a\u7528\u6a21\u5f0f",
        "mode.hint_shift_tab": "  [dim]Shift+Tab[/]       \u2192 \u5728 general / plan / review \u4e4b\u95f4\u5faa\u73af",
        "mode.panel_title": "\u6a21\u5f0f",
        "mode.already_review": "\u5df2\u7ecf\u662f\u5ba1\u9605\u6a21\u5f0f\u3002",
        "mode.already_plan": "\u5df2\u7ecf\u662f\u65b9\u6848\u6a21\u5f0f\u3002",
        "mode.already_general": "\u5df2\u7ecf\u662f\u901a\u7528\u6a21\u5f0f\u3002",
        "mode.back_to_general": "\u2713 \u5df2\u5207\u56de\u901a\u7528\u6a21\u5f0f\u3002",
        "mode.unknown": "\u672a\u77e5\u6a21\u5f0f '{arg}'\u3002\u7528 /mode review | /mode plan | /mode general\u3002",
        "mode.review_panel_title": "\u2691 Review Mode",
        "mode.review_panel_body": (
            "[bold yellow]\u5ba1\u9605\u6a21\u5f0f\u5df2\u5f00\u542f \u2014 \u771f\u6b63\u7684\u9636\u6bb5\u95e8\u63a7[/]\n\n"
            "  \u6a21\u578b\u5199\u4ee3\u7801\u65f6\u5206 3 \u6b65\u8fdb\u884c\uff0c\u6bcf\u6b65 AI \u4f1a[bold]\u8c03\u7528 request_approval \u5de5\u5177[/]\u6682\u505c\u7b49\u4f60\u786e\u8ba4:\n"
            "  \u00b7 Phase 1  [cyan]\u8bbe\u8ba1\u65b9\u6848[/] \u2014 \u5c55\u793a\u51fd\u6570/\u7c7b\u7b7e\u540d\u548c\u804c\u8d23\n"
            "  \u00b7 Phase 2  [blue]\u6570\u636e\u6d41[/]   \u2014 \u6587\u672c\u7bad\u5934\u56fe\u5c55\u793a pipeline\n"
            "  \u00b7 Phase 3  [yellow]\u9010\u6b65\u5b9e\u73b0[/] \u2014 \u4e00\u6b21\u4e00\u4e2a\u51fd\u6570\uff0c\u89e3\u91ca\u6570\u636e\u53d8\u5316\n\n"
            "  \u5ba1\u6279\u9762\u677f\u5f39\u51fa\u65f6\u4f60\u53ef\u4ee5:\n"
            "  \u00b7 [bold]Y / \u7a7a\u56de\u8f66[/] \u2192 \u901a\u8fc7\uff0c\u8fdb\u5165\u4e0b\u4e00\u6b65\n"
            "  \u00b7 [bold]skip / \u5168\u90e8\u5199\u5b8c[/] \u2192 \u901a\u8fc7\u5e76\u8df3\u8fc7\u540e\u7eed\u5ba1\u6279\n"
            "  \u00b7 [bold]N[/] \u7136\u540e\u8f93\u5165\u53cd\u9988 \u2192 \u6253\u56de\u91cd\u5199\n"
            "  \u00b7 \u76f4\u63a5\u8f93\u5165\u4efb\u4f55\u6587\u5b57 \u2192 \u5f53\u4f5c\u53cd\u9988\u5e76\u6253\u56de\n\n"
            "  [dim]Token \u680f\u4f1a\u5b9e\u65f6\u663e\u793a\u5f53\u524d Phase\u3002\u975e\u7f16\u7801\u4efb\u52a1\u4e0d\u53d7\u5f71\u54cd\u3002[/]"
        ),
        "mode.plan_panel_title": "\U0001f4cb Plan Mode",
        "mode.plan_panel_body": (
            "[bold blue]\u65b9\u6848\u6a21\u5f0f\u5df2\u5f00\u542f \u2014 \u53ea\u8bfb\u8c03\u7814 + \u7ed3\u6784\u5316\u65b9\u6848[/]\n\n"
            "  \u8fd9\u4e00\u8f6e AI \u53ea\u4f1a: [cyan]read_file \u00b7 glob \u00b7 tree \u00b7 grep[/]\uff0c\u4e0d\u4f1a\u5199\u4efb\u4f55\u6587\u4ef6\u3002\n"
            "  \u56de\u590d\u672b\u5c3e\u4f1a\u7ed9\u51fa\u7ed3\u6784\u5316\u65b9\u6848\uff08Goal / Files / Steps / Risks\uff09\u3002\n\n"
            "  \u00b7 [bold]/plan approve[/]  \u2192 \u786e\u8ba4\u65b9\u6848\uff0c\u81ea\u52a8\u5207\u5230 acceptEdits \u5f00\u59cb\u52a8\u624b\n"
            "  \u00b7 [bold]/plan cancel[/]   \u2192 \u53d6\u6d88\uff0c\u56de\u5230\u901a\u7528\u6a21\u5f0f\n"
            "  \u00b7 [bold]Shift+Tab[/]       \u2192 \u5faa\u73af\u5207\u6362\u6a21\u5f0f"
        ),

        # ── /plan command ───────────────────────────────────────────────────
        "plan.on_title": "Plan Mode",
        "plan.active_body": (
            "[bold blue]\U0001f4cb Plan Mode \u5f53\u524d\u5f00\u542f[/]\n\n"
            "  \u00b7 [bold]/plan approve[/]  \u2192 \u786e\u8ba4\u65b9\u6848\uff0c\u5207\u5230 acceptEdits \u6267\u884c\n"
            "  \u00b7 [bold]/plan cancel[/]   \u2192 \u53d6\u6d88\uff0c\u56de\u5230\u901a\u7528\u6a21\u5f0f"
        ),
        "plan.inactive_body": (
            "Plan Mode \u672a\u5f00\u542f\u3002\n\n"
            "  \u00b7 [bold]/plan on[/]       \u2192 \u5f00\u542f\uff08\u53ea\u8bfb\u8c03\u7814 + \u7ed3\u6784\u5316\u65b9\u6848\uff09\n"
            "  \u00b7 \u6216 [bold]/mode plan[/]\n"
            "  \u00b7 \u6216 Shift+Tab \u5728\u6a21\u5f0f\u95f4\u5faa\u73af"
        ),
        "plan.off_not_active": "\u65b9\u6848\u6a21\u5f0f\u672a\u5f00\u542f\u3002",
        "plan.cancelled": "\u2713 \u65b9\u6848\u6a21\u5f0f\u5df2\u53d6\u6d88\uff0c\u56de\u5230\u901a\u7528\u6a21\u5f0f\u3002",
        "plan.approve_not_active": "\u8fd8\u6ca1\u5f00\u542f\u65b9\u6848\u6a21\u5f0f \u2014 \u5148 /plan on \u6216 /mode plan\u3002",
        "plan.approved": "\u2713 \u65b9\u6848\u5df2\u6279\u51c6 \u2014 \u5207\u5230 [bold green]acceptEdits[/] \u6a21\u5f0f\u5f00\u59cb\u6267\u884c\u3002",
        "plan.unknown_sub": "\u672a\u77e5\u5b50\u547d\u4ee4 '{sub}'\u3002\u53ef\u9009: on / off / approve / cancel\u3002",

        # ── compact ─────────────────────────────────────────────────────────
        "compact.running": "Compacting context\u2026",
        "compact.title": "\u538b\u7f29\u7ed3\u679c",
        "compact.body": (
            "[green]\u2713 \u538b\u7f29\u5b8c\u6210[/]\n\n"
            "  \u6d88\u606f\u6570 : [white]{mb}[/] \u2192 [white]{ma}[/]\n"
            "  Tokens : [white]{tb:,}[/] \u2192 [white]{ta:,}[/]\n"
            "  \u4fdd\u7559\u6700\u8fd1 [white]{kept}[/] \u6761\u6d88\u606f\u539f\u6587"
        ),
        "compact.failed": "\u538b\u7f29\u5931\u8d25: {err}",

        # ── save / load / resume ───────────────────────────────────────────
        "save.prompt": "  \u4fdd\u5b58\u5230 [{default}] \u203a ",
        "save.done": "\u2713 \u5df2\u4fdd\u5b58 \u2192 {path}",
        "save.transcript_hint": "  (\u4f1a\u8bdd\u4e5f\u5728\u81ea\u52a8\u4fdd\u5b58\u5230 {path})",
        "save.failed": "\u4fdd\u5b58\u5931\u8d25: {err}",
        "load.prompt": "  \u52a0\u8f7d\u6587\u4ef6\u8def\u5f84 \u203a ",
        "load.missing": "\u6587\u4ef6\u4e0d\u5b58\u5728: {path!r}",
        "load.done": "\u2713 \u5df2\u52a0\u8f7d {n} \u6761\u6d88\u606f from {path}",
        "load.failed": "\u52a0\u8f7d\u5931\u8d25: {err}",
        "resume.none": "\u6ca1\u6709\u627e\u5230\u81ea\u52a8\u4fdd\u5b58\u7684\u4f1a\u8bdd\u3002",
        "resume.table_title": "\u6700\u8fd1\u7684\u81ea\u52a8\u4fdd\u5b58\u4f1a\u8bdd",
        "resume.pick_prompt": "  \u8f93\u5165\u7f16\u53f7\u6062\u590d (\u6216 Enter \u53d6\u6d88) \u203a ",
        "resume.range_hint": "\u8bf7\u8f93\u5165 1-{n} \u4e4b\u95f4\u7684\u6570\u5b57",
        "resume.empty_file": "\u4f1a\u8bdd\u6587\u4ef6\u4e3a\u7a7a\u3002",
        "resume.done": "\u2713 \u5df2\u6062\u590d\u4f1a\u8bdd {sid}\u2026 ({n} \u6761\u6d88\u606f)",
        "resume.failed": "\u6062\u590d\u5931\u8d25: {err}",
        "resume.col.session_id": "Session ID",
        "resume.col.messages": "Messages",
        "resume.col.created": "Created",
        "resume.col.last_prompt": "Last prompt",

        # ── reset / config / forget ────────────────────────────────────────
        "reset.done": "\u2713 \u5bf9\u8bdd\u5386\u53f2\u5df2\u6e05\u7a7a\u3002",
        "config.back_to_wizard": "\u8fd4\u56de\u914d\u7f6e\u5411\u5bfc\u2026",
        "forget.done": "\u2713 \u5df2\u5220\u9664\u672c\u5730\u4fdd\u5b58\u7684\u914d\u7f6e ({path})",
        "forget.nothing": "\u6ca1\u6709\u4fdd\u5b58\u7684\u914d\u7f6e\u9700\u8981\u5220\u9664\u3002",

        # ── permissions command ────────────────────────────────────────────
        "perm.current": "  \u5f53\u524d: [bold cyan]{name}[/]\n",
        "perm.label.default": ("default", "\u5e38\u89c4\uff1a\u5199\u64cd\u4f5c\u9010\u6b21\u786e\u8ba4\uff0c\u5371\u9669\u547d\u4ee4\u62d2"),
        "perm.label.accept_edits": ("acceptEdits", "\u63a5\u53d7\u7f16\u8f91\uff1awrite/edit \u81ea\u52a8\u653e\u884c\uff0cbash \u4ecd\u68c0\u67e5"),
        "perm.label.plan": ("plan", "\u8ba1\u5212\u6a21\u5f0f\uff1a\u53ea\u8bfb\u8c03\u7814\uff0cwrite/edit/bash \u5168\u62d2"),
        "perm.label.bypass": ("bypass", "\u7ed5\u8fc7\uff1a\u6240\u6709\u5de5\u5177\u81ea\u52a8\u653e\u884c \u26a0 \u5371\u9669"),
        "perm.panel_title": "\u6743\u9650\u6a21\u5f0f",
        "perm.unknown": "\u672a\u77e5\u6743\u9650\u6a21\u5f0f '{arg}'\u3002/perm \u67e5\u770b\u53ef\u9009\u3002",
        "perm.already": "\u5df2\u7ecf\u662f {name} \u6a21\u5f0f\u3002",
        "perm.bypass_confirm": "  \u26a0 bypass \u6a21\u5f0f\u4f1a\u8df3\u8fc7\u6240\u6709\u6743\u9650\u68c0\u67e5 \u2014 \u786e\u8ba4\u5f00\u542f? [y/N] \u203a ",
        "perm.switched": "\u2713 \u6743\u9650\u6a21\u5f0f\u5207\u5230 [bold]{name}[/] \u2014 {desc}",

        # ── permission prompt (write/risky tool) ──────────────────────────
        "permission.panel_title": "\U0001f512 \u9700\u8981\u786e\u8ba4",
        "permission.panel_body": "  \u5de5\u5177: [bold yellow]{tool}[/]\n{hint}",
        "permission.prompt": "  \u5141\u8bb8\u6267\u884c? [Y/n] \u203a ",

        # ── diff ───────────────────────────────────────────────────────────
        "diff.no_tracker": "\u6587\u4ef6\u8ffd\u8e2a\u5668\u672a\u5c31\u7eea\u3002",
        "diff.no_changes": "\u672c\u6b21\u4f1a\u8bdd\u8fd8\u6ca1\u6709\u6587\u4ef6\u88ab\u5199\u5165\u6216\u7f16\u8f91\u3002",
        "diff.no_match": "\u6ca1\u6709\u5339\u914d '{target}' \u7684\u88ab\u4fee\u6539\u6587\u4ef6\u3002",
        "diff.col.file": "\u6587\u4ef6",
        "diff.col.plus": "+",
        "diff.col.minus": "-",
        "diff.col.writes": "\u5199\u5165\u6b21\u6570",
        "diff.col.status": "\u72b6\u6001",
        "diff.status_new": "\u65b0\u589e",
        "diff.status_edited": "\u4fee\u6539",
        "diff.panel_title": "\u6587\u4ef6\u53d8\u66f4 \u00b7 {n} \u4e2a\u6587\u4ef6 \u00b7 +{added}/-{removed}",

        # ── tasks ──────────────────────────────────────────────────────────
        "tasks.cleared": "\u2713 \u6240\u6709\u540e\u53f0\u4efb\u52a1\u5df2\u6e05\u9664\u3002",
        "tasks.stopped": "\u2713 \u4efb\u52a1 {id} \u5df2\u505c\u6b62\u3002",
        "tasks.stop_missing": "\u4efb\u52a1 {id!r} \u4e0d\u5b58\u5728\u6216\u5df2\u7ed3\u675f\u3002",
        "tasks.not_found": "\u6ca1\u6709\u4efb\u52a1 id = {id!r}",
        "tasks.empty_hint": "\u6ca1\u6709\u540e\u53f0\u4efb\u52a1\u3002\u8ba9 agent \u8c03\u7528 sub_agent(run_in_background=True) \u53ef\u4ee5\u6d3e\u751f\u4e00\u4e2a\u3002",
        "tasks.col.state": "\u72b6\u6001",
        "tasks.col.elapsed": "\u7528\u65f6",
        "tasks.col.description": "\u63cf\u8ff0",
        "tasks.table_title": "\u540e\u53f0\u4efb\u52a1 \u00b7 {n} \u4e2a (running {r})",
        "tasks.footer_hint": "\u63d0\u793a: /tasks <id> \u67e5\u770b\u8f93\u51fa \u00b7 /tasks stop <id> \u505c\u6b62 \u00b7 /tasks clear \u6e05\u7a7a",

        # ── dream ──────────────────────────────────────────────────────────
        "dream.disabled_hint": "Auto-Dream \u672a\u542f\u7528\u3002\u9ed8\u8ba4\u542f\u52a8\u65f6\u4f1a\u5f00\u542f\uff1b\u7528 agent.enable_auto_dream() \u5f00\u3002",
        "dream.off": "\u2713 Auto-Dream \u5df2\u5173\u95ed\u3002",
        "dream.forcing": "\u6b63\u5728\u5f3a\u5236\u6267\u884c\u4e00\u6b21 dream\u2026",
        "dream.forced_done": "\u2713 dream \u5b8c\u6210 \u2014 extractions={e} consolidations={c}",
        "dream.consolidating": "\u6b63\u5728 consolidate\u2026",
        "dream.consolidate_empty": "\u6ca1\u6709\u8db3\u591f\u7684\u6761\u76ee\u53ef\u5408\u5e76\u3002",
        "dream.consolidate_title": "\u6700\u65b0\u5408\u5e76\u6458\u8981",
        "dream.stats_title": "Auto-Dream",
        "dream.stats_state_dreaming": "[yellow]\u25cf dreaming[/]",
        "dream.stats_state_idle": "[green]\u25cf idle[/]",
        "dream.stats_body": (
            "  \u72b6\u6001       : {state}\n"
            "  \u63d0\u53d6\u6b21\u6570   : {e}\n"
            "  \u5408\u5e76\u6b21\u6570   : {c}\n"
            "  \u4e0a\u6b21\u8fd0\u884c   : {last}\n"
            "  \u4e0a\u6b21\u9519\u8bef   : {err}\n"
            "  \u8bb0\u5fc6\u6761\u76ee   : {mem}\n\n"
            "  \u95f4\u9694 (\u8f6e)  : {interval}\n"
            "  \u5408\u5e76\u5468\u671f   : \u6bcf {cons} \u6b21\u63d0\u53d6\n\n"
            "  [dim]/dream force[/]        \u5f3a\u5236\u6267\u884c\u4e00\u8f6e\n"
            "  [dim]/dream consolidate[/]  \u53ea\u8dd1\u5408\u5e76\n"
            "  [dim]/dream off[/]          \u5173\u95ed"
        ),

        # ── mcp ────────────────────────────────────────────────────────────
        "mcp.no_server": "\u6ca1\u6709 MCP \u670d\u52a1\u5668 {name!r}",
        "mcp.reconnect_ok": "\u2713 {name} \u91cd\u8fde\u6210\u529f \u2014 {n} \u4e2a\u5de5\u5177",
        "mcp.reconnect_failed": "\u2717 {name} \u91cd\u8fde\u5931\u8d25: {err}",
        "mcp.disconnected": "\u2713 {name} \u5df2\u65ad\u5f00\u3002",
        "mcp.not_connected": "{name!r} \u672a\u8fde\u63a5\u3002",
        "mcp.empty_body": (
            "\u5f53\u524d\u6ca1\u6709 MCP \u670d\u52a1\u5668\u8fde\u63a5\u3002\n\n"
            "\u8981\u542f\u7528: \u7f16\u8f91 [cyan]{path}[/], \u683c\u5f0f:\n\n"
            "[dim]{{\n"
            '  "mcpServers": {{\n'
            '    "filesystem": {{\n'
            '      "command": "npx",\n'
            '      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]\n'
            "    }}\n"
            "  }}\n"
            "}}[/]\n\n"
            "\u7136\u540e /mcp reconnect <name> \u6216\u91cd\u542f lumen\u3002"
        ),
        "mcp.col.status": "\u72b6\u6001",
        "mcp.col.tools": "\u5de5\u5177\u6570",
        "mcp.col.detail": "\u9519\u8bef / \u547d\u4ee4",
        "mcp.table_title": "MCP \u670d\u52a1\u5668 \u00b7 {n} \u4e2a",
        "mcp.footer_hint": "\u63d0\u793a: /mcp reconnect <name> \u91cd\u8fde \u00b7 /mcp disconnect <name> \u65ad\u5f00",

        # ── vim ────────────────────────────────────────────────────────────
        "vim.off": "\u2713 Vim \u6a21\u5f0f\u5173\u95ed (\u56de\u5230 Emacs/\u9ed8\u8ba4)",
        "vim.on": "\u2713 Vim \u6a21\u5f0f\u5f00\u542f  \u00b7  Esc\u2192Normal  \u00b7  i\u2192Insert  \u00b7  :\u2192\u5192\u53f7\u547d\u4ee4",

        # ── /lang ─────────────────────────────────────────────────────────
        "lang.current": "\u5f53\u524d\u8bed\u8a00: [bold cyan]{name}[/]",
        "lang.switched": "\u2713 \u8bed\u8a00\u5df2\u5207\u6362\u4e3a [bold cyan]{name}[/]",
        "lang.already": "\u5df2\u7ecf\u662f [bold cyan]{name}[/]\u3002",
        "lang.unknown": "\u672a\u77e5\u8bed\u8a00 '{arg}'\u3002\u53ef\u9009: en | zh\u3002",
        "lang.available": "\u53ef\u9009: [bold]/lang en[/] \u00b7 [bold]/lang zh[/]",
        "lang.name.en": "English",
        "lang.name.zh": "\u4e2d\u6587",

        # ── slash command catalog (/help descriptions) ─────────────────────
        "cmd.help": "\u663e\u793a\u5e2e\u52a9",
        "cmd.status": "Token \u7528\u91cf\u548c\u4f1a\u8bdd\u4fe1\u606f",
        "cmd.mode": "\u5207\u6362\u6a21\u5f0f (general | plan | review)",
        "cmd.plan": "Plan Mode \u63a7\u5236 (on/off/approve/cancel)",
        "cmd.diff": "\u5c55\u793a\u672c\u6b21\u4f1a\u8bdd agent \u4fee\u6539\u8fc7\u7684\u6587\u4ef6 (\u53ef\u5e26\u8def\u5f84\u53c2\u6570)",
        "cmd.tasks": "\u540e\u53f0\u4efb\u52a1 (\u5217\u8868 / \u67e5\u770b id / stop id / clear)",
        "cmd.mcp": "MCP \u670d\u52a1\u5668 (list / reconnect / disconnect)",
        "cmd.dream": "Auto-Dream \u8bb0\u5fc6\u5408\u5e76 (force/consolidate/off)",
        "cmd.perm": "\u6743\u9650\u6a21\u5f0f (default/acceptEdits/plan/bypass)",
        "cmd.compact": "\u624b\u52a8\u538b\u7f29\u4e0a\u4e0b\u6587",
        "cmd.reset": "\u6e05\u7a7a\u5bf9\u8bdd\u5386\u53f2",
        "cmd.save": "\u4fdd\u5b58\u4f1a\u8bdd\u5230\u6587\u4ef6",
        "cmd.load": "\u4ece\u6587\u4ef6\u52a0\u8f7d\u4f1a\u8bdd",
        "cmd.resume": "\u6062\u590d\u6700\u8fd1\u7684\u81ea\u52a8\u4fdd\u5b58\u4f1a\u8bdd",
        "cmd.config": "\u91cd\u65b0\u914d\u7f6e\u6a21\u578b\u548c Key",
        "cmd.forget": "\u5220\u9664\u672c\u5730\u4fdd\u5b58\u7684 API Key",
        "cmd.vim": "\u5207\u6362 Vim \u6a21\u5f0f",
        "cmd.lang": "\u5207\u6362\u754c\u9762\u8bed\u8a00 (en | zh)",
        "cmd.quit": "\u9000\u51fa",

        # ── command subarg menu labels ─────────────────────────────────────
        "cmd.sub.mode.general": "\u901a\u7528\u6a21\u5f0f",
        "cmd.sub.mode.plan": "\u65b9\u6848\u6a21\u5f0f\uff08\u53ea\u8bfb\u8c03\u7814\uff09",
        "cmd.sub.mode.review": "\u5ba1\u9605\u6a21\u5f0f\uff08\u9636\u6bb5\u95e8\u63a7\uff09",
        "cmd.sub.plan.on": "\u5f00\u542f\u65b9\u6848\u6a21\u5f0f",
        "cmd.sub.plan.approve": "\u6279\u51c6\u65b9\u6848\u5e76\u5f00\u59cb\u6267\u884c",
        "cmd.sub.plan.cancel": "\u53d6\u6d88\u65b9\u6848\u6a21\u5f0f",
        "cmd.sub.plan.off": "\u540c cancel",
        "cmd.sub.tasks.clear": "\u6e05\u7a7a\u6240\u6709\u4efb\u52a1",
        "cmd.sub.tasks.stop": "/tasks stop <id> \u505c\u6b62\u4efb\u52a1",
        "cmd.sub.mcp.reconnect": "/mcp reconnect <name>",
        "cmd.sub.mcp.disconnect": "/mcp disconnect <name>",
        "cmd.sub.dream.force": "\u5f3a\u5236\u8dd1\u4e00\u8f6e",
        "cmd.sub.dream.consolidate": "\u53ea\u8dd1\u5408\u5e76",
        "cmd.sub.dream.off": "\u5173\u95ed Auto-Dream",
        "cmd.sub.perm.default": "\u5199\u64cd\u4f5c\u9010\u6b21\u786e\u8ba4",
        "cmd.sub.perm.accept_edits": "write/edit \u81ea\u52a8\u653e\u884c",
        "cmd.sub.perm.plan": "\u53ea\u8bfb\u8c03\u7814\u6a21\u5f0f",
        "cmd.sub.perm.bypass": "\u26a0 \u5168\u90e8\u653e\u884c",
        "cmd.sub.lang.en": "\u5207\u6362\u5230\u82f1\u6587",
        "cmd.sub.lang.zh": "\u5207\u6362\u5230\u4e2d\u6587",

        # ── bottom toolbar ─────────────────────────────────────────────────
        "toolbar.review_idle": "\u5f85\u547d",
        "toolbar.review_design": "\u8bbe\u8ba1",
        "toolbar.review_pipeline": "\u6570\u636e\u6d41",
        "toolbar.review_impl": "\u5b9e\u73b0\u00d7{n}",
        "toolbar.review_complete": "\u5b8c\u6210",
        "toolbar.mode_review": " \u2691 review \u00b7 {phase} ",
        "toolbar.mode_plan": " \U0001f4cb plan ",
        "toolbar.mode_general": " general ",
        "toolbar.perm_accept_edits": " \u270e accept-edits \u2502",
        "toolbar.perm_plan": " \U0001f4cb plan-only \u2502",
        "toolbar.perm_bypass": " \u26a0 BYPASS \u2502",
        "toolbar.line": (
            " mode:{mode}\u2502{perm} \u21e7\u21e5 \u5207\u6a21\u5f0f \u2502 / \u547d\u4ee4 \u2502 @ \u6587\u4ef6 \u2502 "
            "\u21e7\u23ce / \u2325\u23ce \u6362\u884c \u2502 {model} "
        ),

        # ── generic messages ───────────────────────────────────────────────
        "msg.unknown_command": "\u672a\u77e5\u547d\u4ee4: {cmd}  \u8f93\u5165 /help \u67e5\u770b\u5e2e\u52a9",
        "msg.cmd_failed": "{cmd} \u6267\u884c\u5931\u8d25: {err}",
        "msg.agent_failed": "Agent \u521b\u5efa\u5931\u8d25: {err}",
        "msg.cancel_reply": "\u5df2\u53d6\u6d88\u3002\uff08\u518d\u6309\u4e00\u6b21 Ctrl+C \u9000\u51fa\uff09",
        "msg.user_cmd_load_failed": "  \u26a0 \u7528\u6237\u547d\u4ee4 {file} \u52a0\u8f7d\u5931\u8d25: {err}",
        "msg.user_cmd_loaded": "\u2713 \u52a0\u8f7d\u7528\u6237\u547d\u4ee4: {names}",
        "msg.reply_done": "\u56de\u590d\u5b8c\u6210 \u00b7 {preview}",
        "msg.thinking": "  [dim]\u6b63\u5728\u601d\u8003\u2026[/]",
        "msg.lumen_thinking": "  Lumen  \u25cf  thinking\u2026",
        "msg.lumen_streaming": "  Lumen  \u25cf  streaming\u2026",
        "msg.lumen": "  Lumen",
        "msg.you": "  You",

        # ── approval handler ───────────────────────────────────────────────
        "approval.prompt": "\u56de\u8f66=\u901a\u8fc7  n=\u9a73\u56de  skip=\u5168\u90e8\u901a\u8fc7  (\u6216\u76f4\u63a5\u8f93\u5165\u53cd\u9988) \u203a ",
        "approval.feedback_prompt": "\u53cd\u9988 \u203a ",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_language(value: str | None) -> str | None:
    """Accept common aliases (English, 英文, CN, zh_CN, …) and map to en/zh."""
    if not value:
        return None
    v = value.strip().lower().replace("_", "-")
    if v in ("en", "eng", "english", "英文", "英语", "en-us", "en-gb"):
        return "en"
    if v in ("zh", "cn", "zh-cn", "zh-hans", "zh-tw", "chinese", "中文", "中"):
        return "zh"
    return None


def _detect_initial_language() -> str:
    """Pick initial language from env if the user has not chosen one yet.

    Priority: LUMEN_LANG > LANG (POSIX locale) > DEFAULT_LANGUAGE.
    """
    env = os.environ.get("LUMEN_LANG") or os.environ.get("LANG", "")
    norm = normalize_language(env.split(".", 1)[0])
    return norm or DEFAULT_LANGUAGE


def get_language() -> str:
    return _current


def set_language(lang: str) -> bool:
    """Switch the active language. Returns True on success."""
    global _current
    norm = normalize_language(lang)
    if norm is None or norm not in TRANSLATIONS:
        return False
    _current = norm
    return True


def language_display_name(lang: str | None = None) -> str:
    """Human-readable name of a language, in the current UI language."""
    code = lang or _current
    return t(f"lang.name.{code}")


def t(key: str, **kwargs: Any) -> str:
    """Translate a key. Falls back `zh` → `en` → raw key. Never raises."""
    for lang in (_current, DEFAULT_LANGUAGE):
        table = TRANSLATIONS.get(lang)
        if table is None:
            continue
        value = table.get(key)
        if value is not None:
            if kwargs:
                try:
                    return value.format(**kwargs) if isinstance(value, str) else value
                except (KeyError, IndexError, ValueError):
                    return value if isinstance(value, str) else key
            return value if isinstance(value, str) else key
    return key


def tt(key: str) -> tuple[str, str]:
    """Fetch a key whose value is a 2-tuple (name, description) — used by the
    permission table. Falls back to English, then ("", "") if truly missing."""
    for lang in (_current, DEFAULT_LANGUAGE):
        table = TRANSLATIONS.get(lang)
        if table is None:
            continue
        value = table.get(key)
        if isinstance(value, tuple) and len(value) == 2:
            return value
    return ("", "")


# Seed from env on first import. chat.py will overwrite this once config.json
# is loaded (user's explicit choice wins over the env guess).
set_language(_detect_initial_language())
