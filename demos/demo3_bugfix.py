"""
Demo 3: Agent 找 Bug 并修复

先写一个有 bug 的文件，让 Agent 找到并修复它。
展示：read_file → 分析 → edit_file → bash 验证 的完整 debug 流程。
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lumen import Agent, PermissionChecker
from lumen.tools import FileReadTool, FileWriteTool, FileEditTool, BashTool

API_KEY = os.environ.get("GETGOAPI_API_KEY", "sk-your-key-here")
BASE_URL = "https://api.getgoapi.com/v1"
MODEL = "gpt-4.1-mini-2025-04-14"

BUG_FILE = "/tmp/lumen_demo/buggy_calc.py"

# 故意写一个有 3 个 bug 的文件
BUGGY_CODE = '''\
"""简单计算器 — 有几个 bug，你能找到吗？"""

def add(a, b):
    return a - b  # Bug 1: 应该是 +

def multiply(a, b):
    return a * a  # Bug 2: 应该是 a * b

def divide(a, b):
    return a / b  # Bug 3: 没有处理除零

def power(a, n):
    """计算 a 的 n 次方"""
    result = 0  # Bug 4: 应该初始化为 1
    for _ in range(n):
        result *= a
    return result


if __name__ == "__main__":
    # 测试
    assert add(2, 3) == 5, f"add failed: {add(2, 3)}"
    assert multiply(3, 4) == 12, f"multiply failed: {multiply(3, 4)}"
    assert divide(10, 2) == 5, f"divide failed: {divide(10, 2)}"
    assert divide(10, 0) == float("inf"), f"divide by zero failed"
    assert power(2, 3) == 8, f"power failed: {power(2, 3)}"
    print("All tests passed!")
'''


async def main():
    Path("/tmp/lumen_demo").mkdir(exist_ok=True)
    Path(BUG_FILE).write_text(BUGGY_CODE)
    print("=" * 60)
    print("  Demo 3: Agent 找 Bug 并修复")
    print("=" * 60)
    print(f"  📄 已创建有 bug 的文件: {BUG_FILE}")
    print(f"  🐛 内含 4 个 bug，看 Agent 能不能全部修复\n")

    agent = Agent(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        inject_git_state=False,
        inject_memory=False,
        permission_checker=PermissionChecker(ask_fn=lambda n, i: True),
        tools=[FileReadTool(), FileEditTool(), BashTool()],
    )

    calls = []
    def on_tool(name, args):
        calls.append(name)
        if name == "edit_file":
            old = args.get("old_string", "")[:40]
            new = args.get("new_string", "")[:40]
            print(f"  ✏️  edit: '{old}' → '{new}'")
        elif name == "bash":
            print(f"  💻 bash: {args.get('command', '')[:60]}")
        else:
            print(f"  🔧 {name}")

    def on_result(name, output, is_error):
        status = "❌" if is_error else "✅"
        first = output.strip().split("\n")[0][:60]
        print(f"    {status} {first}")

    response = await agent.chat(
        f"请读取 {BUG_FILE}，找出所有 bug 并用 edit_file 逐个修复。"
        f"修完后用 bash 执行 python {BUG_FILE} 验证所有测试通过。",
        on_tool_call=on_tool,
        on_tool_result=on_result,
    )

    print(f"\n{'─' * 60}")
    print(f"📋 调用: {' → '.join(calls)}")
    print(f"📊 Token: {agent.token_usage.total}")
    print(f"{'─' * 60}")
    print(response.content)

    # Verify
    fixed = Path(BUG_FILE).read_text()
    checks = [
        ("a + b" in fixed or "a+b" in fixed, "add 修复"),
        ("a * b" in fixed or "a*b" in fixed, "multiply 修复"),
        ("result = 1" in fixed, "power 初始值修复"),
    ]
    print(f"\n🔍 验证修复:")
    for passed, desc in checks:
        print(f"  {'✅' if passed else '❌'} {desc}")


if __name__ == "__main__":
    asyncio.run(main())
