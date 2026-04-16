"""
Demo 2: Agent 写代码并执行

让 Agent 写一个小程序，然后用 bash 执行验证。
展示：write_file → bash 的完整写+跑流程。
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lumen import Agent, PermissionChecker
from lumen.tools import FileWriteTool, FileReadTool, BashTool, FileEditTool

API_KEY = os.environ.get("GETGOAPI_API_KEY", "sk-your-key-here")
BASE_URL = "https://api.getgoapi.com/v1"
MODEL = "gpt-4.1-mini-2025-04-14"

OUTPUT_DIR = "/tmp/lumen_demo"


async def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    agent = Agent(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        inject_git_state=False,
        inject_memory=False,
        permission_checker=PermissionChecker(ask_fn=lambda n, i: True),
        tools=[FileWriteTool(), FileReadTool(), FileEditTool(), BashTool()],
    )

    print("=" * 60)
    print("  Demo 2: Agent 写代码并执行")
    print("=" * 60)

    calls = []
    def on_tool(name, args):
        hint = args.get("file_path") or args.get("command", "")
        calls.append(name)
        print(f"  🔧 {name}  {str(hint)[:70]}")

    def on_result(name, output, is_error):
        status = "❌" if is_error else "✅"
        preview = output.strip().split("\n")[0][:70]
        print(f"    {status} {preview}")

    response = await agent.chat(
        f"请在 {OUTPUT_DIR}/snake_game.py 写一个终端贪吃蛇的核心逻辑（不需要真的跑游戏循环，"
        f"写一个 Snake 类，有 move/eat/is_dead 方法，加上简单的单元测试）。"
        f"然后用 bash 执行 python {OUTPUT_DIR}/snake_game.py 验证测试通过。",
        on_tool_call=on_tool,
        on_tool_result=on_result,
    )

    print(f"\n{'─' * 60}")
    print(f"📋 调用: {' → '.join(calls)}")
    print(f"📊 Token: {agent.token_usage.total}")
    print(f"{'─' * 60}")
    print(response.content)

    # Show what was written
    out = Path(f"{OUTPUT_DIR}/snake_game.py")
    if out.exists():
        lines = out.read_text().count("\n")
        print(f"\n📄 生成文件: {out}  ({lines} 行)")


if __name__ == "__main__":
    asyncio.run(main())
