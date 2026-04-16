"""
Demo 4: 多模型格式适配 — 同一套工具，不同模型自动适配

展示 lumen 的核心能力：
  同一个 Tool 定义 → 自动适配不同模型的工具格式 → 都能正确调用
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lumen import Agent, PermissionChecker
from lumen.providers.model_profiles import detect_profile, convert_tool_schema
from lumen.tools import BashTool, TreeTool

API_KEY = os.environ.get("GETGOAPI_API_KEY", "sk-your-key-here")
BASE_URL = "https://api.getgoapi.com/v1"

# 两个不同家族的模型用相同的 prompt + tools
MODELS = [
    ("gpt-4.1-nano-2025-04-14", "最便宜的 GPT"),
    ("gpt-4.1-mini-2025-04-14", "中等 GPT"),
]

PROMPT = "用 bash 执行 python3 -c \"print(sum(range(101)))\"，告诉我 0 到 100 的累加和是多少。"


async def main():
    print("=" * 60)
    print("  Demo 4: 多模型格式适配")
    print("=" * 60)

    # 1. 展示格式差异
    print("\n  📐 同一个工具在不同模型下的 Schema 格式:\n")
    tool_def = {
        "name": "bash",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"},
            },
            "required": ["command"],
        },
    }

    for model_name in ["gpt-4o", "claude-sonnet-4-6", "deepseek-chat", "llama3.1"]:
        profile = detect_profile(model_name)
        schema = convert_tool_schema(tool_def, profile)

        # 只显示关键差异
        if "function" in schema:
            fmt = f"type=function, function.name={schema['function']['name']}"
        elif "input_schema" in schema:
            fmt = f"name={schema['name']}, has input_schema"
        else:
            fmt = str(list(schema.keys()))

        tokens = f"tokens={profile.capabilities.tool_token_prefix}" if profile.capabilities.native_tool_tokens else ""
        print(f"    {model_name:<25} → {profile.family:<10} {fmt}  {tokens}")

    # 2. 实际调用
    print(f"\n  🚀 用不同模型执行同一任务:\n")
    print(f"    Prompt: {PROMPT}\n")

    perms = PermissionChecker(ask_fn=lambda n, i: True)

    for model, desc in MODELS:
        profile = detect_profile(model)
        agent = Agent(
            api_key=API_KEY,
            model=model,
            base_url=BASE_URL,
            inject_git_state=False,
            inject_memory=False,
            permission_checker=perms,
            tools=[BashTool()],
        )

        tool_used = ""
        def on_tool(name, args, _m=model):
            nonlocal tool_used
            tool_used = f"{name}({args.get('command', '')[:30]})"

        try:
            resp = await agent.chat(PROMPT, on_tool_call=on_tool)
            answer = resp.content.strip().split("\n")[0][:60] if resp.content else "(empty)"
            has_5050 = "5050" in (resp.content or "")
            icon = "✅" if has_5050 else "⚠️"
            print(f"    {icon} {model}")
            print(f"       Profile: {profile.family} ({profile.display_name})")
            print(f"       Tool:    {tool_used}")
            print(f"       Answer:  {answer}")
            print(f"       Tokens:  {resp.token_usage.total}")
            print()
        except Exception as e:
            print(f"    ❌ {model}: {str(e)[:60]}\n")


if __name__ == "__main__":
    asyncio.run(main())
