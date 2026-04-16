"""
Demo 1: Agent 自主阅读代码库

给 Agent 一个问题，让它自己用工具探索代码库并回答。
展示：tree → glob → read_file 的自主调用链。
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lumen import Agent, PermissionChecker
from lumen.tools import TreeTool, FileReadTool, GlobTool, GrepTool, DefinitionsTool

API_KEY = os.environ.get("GETGOAPI_API_KEY", "sk-your-key-here")
BASE_URL = "https://api.getgoapi.com/v1"
MODEL = "gpt-4.1-mini-2025-04-14"


async def main():
    agent = Agent(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        inject_git_state=False,
        inject_memory=False,
        permission_checker=PermissionChecker(ask_fn=lambda n, i: True),
        tools=[TreeTool(), FileReadTool(), GlobTool(), GrepTool(), DefinitionsTool()],
    )

    print("=" * 60)
    print("  Demo 1: Agent 自主阅读代码库")
    print("=" * 60)

    calls = []
    def on_tool(name, args):
        hint = args.get("file_path") or args.get("path") or args.get("pattern") or ""
        calls.append(f"{name}({hint})")
        print(f"  🔧 {name}  {str(hint)[:60]}")

    response = await agent.chat(
        "阅读当前项目的 lumen/lumen/services/ 目录。"
        "告诉我这个目录里有哪些模块，每个模块的核心功能是什么？用中文简洁回答。",
        on_tool_call=on_tool,
    )

    print(f"\n{'─' * 60}")
    print(f"📋 工具调用链: {' → '.join(calls)}")
    print(f"📊 Token: {agent.token_usage.total}")
    print(f"{'─' * 60}")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
