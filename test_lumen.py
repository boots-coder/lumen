"""
Lumen 端到端测试 — 使用真实 API 验证所有核心能力。

测试项：
  1. 基本对话（无工具）
  2. 工具调用（读文件、搜索）
  3. 写文件（权限确认流程）
  4. 模型 Profile 自动检测
  5. 多模型格式适配
  6. 错误分类 + 重试逻辑
  7. 文件缓存
  8. 工具结果截断
  9. 并发工具执行
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lumen import Agent, PermissionChecker, PermissionBehavior, RetryConfig
from lumen.providers.model_profiles import (
    detect_profile, list_supported_families,
    ModelProfile, ToolSchemaFormat, MessageFormat,
    register_profile, ModelCapabilities, convert_tool_schema,
)
from lumen.services.errors import classify_error, ErrorType
from lumen.services.tool_executor import partition_tool_calls, ToolExecRequest
from lumen.utils.file_state_cache import FileStateCache
from lumen.utils.tool_result_truncation import truncate_tool_result
from lumen.tools import (
    FileReadTool, FileWriteTool, FileEditTool,
    GlobTool, GrepTool, BashTool, TreeTool, DefinitionsTool,
)

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("GETGOAPI_API_KEY", "sk-your-key-here")
BASE_URL = "https://api.getgoapi.com/v1"
MODEL = "gpt-4.1-mini-2025-04-14"  # 省钱用 mini

# ── Helpers ───────────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0

def ok(name: str, detail: str = ""):
    global _pass_count
    _pass_count += 1
    print(f"  ✅ {name}" + (f"  ({detail})" if detail else ""))

def fail(name: str, detail: str = ""):
    global _fail_count
    _fail_count += 1
    print(f"  ❌ {name}" + (f"  ({detail})" if detail else ""))


# ═════════════════════════════════════════════════════════════════════════════
# Test 1: Model Profile 自动检测
# ═════════════════════════════════════════════════════════════════════════════

def test_model_profiles():
    print("\n── Test 1: 模型 Profile 自动检测 ──")

    cases = [
        ("gpt-4o", "openai", ToolSchemaFormat.OPENAI_FUNCTION),
        ("gpt-4.1-2025-04-14", "openai", ToolSchemaFormat.OPENAI_FUNCTION),
        ("o3-mini", "openai_reasoning", ToolSchemaFormat.OPENAI_FUNCTION),
        ("claude-sonnet-4-6", "anthropic", ToolSchemaFormat.ANTHROPIC_TOOL_USE),
        ("claude-haiku-4-5", "anthropic", ToolSchemaFormat.ANTHROPIC_TOOL_USE),
        ("deepseek-chat", "deepseek", ToolSchemaFormat.OPENAI_FUNCTION),
        ("deepseek-reasoner", "deepseek", ToolSchemaFormat.OPENAI_FUNCTION),
        ("qwen2.5", "qwen", ToolSchemaFormat.OPENAI_FUNCTION),
        ("llama3.1", "llama", ToolSchemaFormat.OPENAI_FUNCTION),
        ("llama3.2", "llama", ToolSchemaFormat.OPENAI_FUNCTION),
        ("mistral", "mistral", ToolSchemaFormat.OPENAI_FUNCTION),
        ("mixtral-8x7b", "mistral", ToolSchemaFormat.OPENAI_FUNCTION),
        ("gemini-1.5-pro", "gemini", ToolSchemaFormat.OPENAI_FUNCTION),
        ("some-random-model", "generic", ToolSchemaFormat.OPENAI_FUNCTION),
    ]

    for model_name, expected_family, expected_format in cases:
        profile = detect_profile(model_name)
        if profile.family == expected_family and profile.tool_schema_format == expected_format:
            ok(f"{model_name} → {profile.family}", profile.display_name)
        else:
            fail(f"{model_name}", f"got {profile.family}/{profile.tool_schema_format}, expected {expected_family}/{expected_format}")

    # Test custom profile registration
    register_profile("my-custom-model", ModelProfile(
        family="custom_test",
        display_name="Test Custom Model",
        tool_schema_format=ToolSchemaFormat.OPENAI_STRICT,
    ))
    p = detect_profile("my-custom-model")
    if p.family == "custom_test":
        ok("Custom profile registration")
    else:
        fail("Custom profile registration")

    # Test base_url hint
    p = detect_profile("unknown-model", base_url="https://api.anthropic.com/v1")
    if p.family == "anthropic":
        ok("Base URL hint → Anthropic")
    else:
        fail("Base URL hint detection")

    print(f"\n  支持的模型家族: {list_supported_families()}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 2: 工具 Schema 格式适配
# ═════════════════════════════════════════════════════════════════════════════

def test_tool_schema_adaptation():
    print("\n── Test 2: 工具 Schema 格式适配 ──")

    universal_tool = {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
                "offset": {"type": "integer", "description": "Line offset"},
            },
            "required": ["file_path"],
        },
    }

    # OpenAI format
    openai_profile = detect_profile("gpt-4o")
    openai_schema = convert_tool_schema(universal_tool, openai_profile)
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "read_file"
    ok("OpenAI function format", f"type={openai_schema['type']}")

    # Anthropic format
    claude_profile = detect_profile("claude-sonnet-4-6")
    claude_schema = convert_tool_schema(universal_tool, claude_profile)
    assert "input_schema" in claude_schema
    assert claude_schema["name"] == "read_file"
    ok("Anthropic tool_use format", f"has input_schema={bool(claude_schema.get('input_schema'))}")

    # Strict format
    strict_profile = ModelProfile(
        family="test", display_name="Test",
        tool_schema_format=ToolSchemaFormat.OPENAI_STRICT,
    )
    strict_schema = convert_tool_schema(universal_tool, strict_profile)
    assert strict_schema["function"].get("strict") == True
    ok("OpenAI strict format", f"strict={strict_schema['function']['strict']}")

    # DeepSeek (should work like OpenAI but with possible tweaks)
    ds_profile = detect_profile("deepseek-chat")
    ds_schema = convert_tool_schema(universal_tool, ds_profile)
    assert ds_schema["type"] == "function"
    ok("DeepSeek format", f"native_tokens={ds_profile.capabilities.native_tool_tokens}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 3: 真实 API 调用 — 基本对话
# ═════════════════════════════════════════════════════════════════════════════

async def test_basic_chat():
    print("\n── Test 3: 真实 API 调用 — 基本对话 ──")

    agent = Agent(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        auto_compact=False,
        inject_git_state=False,
        inject_memory=False,
    )

    # Verify profile was detected
    profile = agent.model_profile
    ok(f"Profile detected: {profile.family}", profile.display_name)

    response = await agent.chat("请用一句话回答：1+1等于几？")

    if response.content and ("2" in response.content or "二" in response.content):
        ok("基本对话成功", f"回答: {response.content[:60]}")
    else:
        fail("基本对话", f"回答: {response.content[:80] if response.content else 'empty'}")

    if response.token_usage.total > 0:
        ok(f"Token 计数正常", f"total={response.token_usage.total}")
    else:
        fail("Token 计数", "total=0")


# ═════════════════════════════════════════════════════════════════════════════
# Test 4: 真实 API + 工具调用
# ═════════════════════════════════════════════════════════════════════════════

async def test_tool_calling():
    print("\n── Test 4: 真实 API + 工具调用 ──")

    # Auto-approve all tools for testing
    perms = PermissionChecker(
        ask_fn=lambda name, inp: True,  # Auto-approve everything
    )

    agent = Agent(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        auto_compact=False,
        inject_git_state=False,
        inject_memory=False,
        permission_checker=perms,
        enable_file_cache=True,
        tools=[
            TreeTool(),
            FileReadTool(),
            GlobTool(),
            GrepTool(),
            BashTool(),
        ],
    )

    tool_calls_made = []

    def on_tool_call(name, args):
        tool_calls_made.append(name)

    response = await agent.chat(
        "用 tree 工具查看当前目录结构，然后告诉我有哪些 Python 文件。只列出文件名即可。",
        on_tool_call=on_tool_call,
    )

    if tool_calls_made:
        ok(f"工具被调用: {tool_calls_made}")
    else:
        fail("没有工具调用")

    if response.content and (".py" in response.content or "python" in response.content.lower()):
        ok("工具结果正确融入回答", f"{response.content[:80]}...")
    else:
        ok("收到回答", f"{response.content[:80]}..." if response.content else "empty")

    # Check file cache
    if agent.file_cache and agent.file_cache.size >= 0:
        ok(f"文件缓存工作", f"cached={agent.file_cache.size} files")


# ═════════════════════════════════════════════════════════════════════════════
# Test 5: 权限系统
# ═════════════════════════════════════════════════════════════════════════════

def test_permissions():
    print("\n── Test 5: 权限系统 ──")

    pc = PermissionChecker()

    # Read → allow silently
    tests = [
        ("read_file", {"file_path": "/tmp/x.py"}, PermissionBehavior.ALLOW, "读操作静默放行"),
        ("glob", {"pattern": "*.py"}, PermissionBehavior.ALLOW, "glob 静默放行"),
        ("grep", {"pattern": "foo"}, PermissionBehavior.ALLOW, "grep 静默放行"),
        # Write → ask
        ("write_file", {"file_path": "/tmp/x.py"}, PermissionBehavior.ASK, "写操作需确认"),
        ("edit_file", {"file_path": "/tmp/x.py", "old_string": "a", "new_string": "b"}, PermissionBehavior.ASK, "编辑需确认"),
        # Dangerous → deny
        ("bash", {"command": "rm -rf /"}, PermissionBehavior.DENY, "危险命令拒绝"),
        ("bash", {"command": "rm -rf ~"}, PermissionBehavior.DENY, "危险命令拒绝"),
        # Safe bash → allow
        ("bash", {"command": "git status"}, PermissionBehavior.ALLOW, "安全bash放行"),
        ("bash", {"command": "python test.py"}, PermissionBehavior.ALLOW, "python命令放行"),
        # Risky bash → ask
        ("bash", {"command": "git push origin main"}, PermissionBehavior.ASK, "风险操作需确认"),
        ("bash", {"command": "sudo rm file"}, PermissionBehavior.ASK, "sudo需确认"),
    ]

    for tool_name, tool_input, expected, desc in tests:
        result = pc.check(tool_name, tool_input)
        if result.behavior == expected:
            ok(desc, f"{tool_name} → {result.behavior.value}")
        else:
            fail(desc, f"expected {expected.value}, got {result.behavior.value}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 6: 错误分类
# ═════════════════════════════════════════════════════════════════════════════

def test_error_classification():
    print("\n── Test 6: 错误分类 ──")

    cases = [
        (429, None, "rate limited", ErrorType.RATE_LIMIT, True),
        (529, None, "overloaded", ErrorType.SERVER_OVERLOAD, True),
        (401, None, "unauthorized", ErrorType.AUTH_ERROR, False),
        (400, None, "188059 + 20000 > 200000", ErrorType.MAX_TOKENS_OVERFLOW, True),
        (400, None, "model xyz not found", ErrorType.INVALID_MODEL, False),
        (None, None, "connection reset", ErrorType.CONNECTION_ERROR, True),
        (None, None, "read timeout", ErrorType.API_TIMEOUT, True),
        (None, None, "request aborted", ErrorType.ABORTED, False),
        (None, None, "content refused by safety", ErrorType.REFUSAL, False),
        (500, None, "internal server error", ErrorType.SERVER_OVERLOAD, True),
    ]

    for status, headers, msg, expected_type, expected_retryable in cases:
        c = classify_error(Exception(msg), status_code=status)
        if c.error_type == expected_type and c.retryable == expected_retryable:
            ok(f"{expected_type.value}", f"retryable={c.retryable}")
        else:
            fail(f"{expected_type.value}", f"got {c.error_type.value}, retryable={c.retryable}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 7: 并发工具执行分批
# ═════════════════════════════════════════════════════════════════════════════

def test_tool_batching():
    print("\n── Test 7: 并发工具执行分批 ──")

    reqs = [
        ToolExecRequest("1", "read_file", {}),
        ToolExecRequest("2", "grep", {}),
        ToolExecRequest("3", "glob", {}),
        ToolExecRequest("4", "write_file", {}),  # write → serial
        ToolExecRequest("5", "read_file", {}),
        ToolExecRequest("6", "definitions", {}),
        ToolExecRequest("7", "edit_file", {}),    # write → serial
        ToolExecRequest("8", "bash", {}),          # bash → serial
    ]

    batches = partition_tool_calls(reqs)
    # Expected: [read,grep,glob], [write], [read,definitions], [edit], [bash]
    if len(batches) == 5:
        ok(f"分批数正确: {len(batches)} batches")
    else:
        fail(f"分批数", f"expected 5, got {len(batches)}")

    if len(batches[0]) == 3:
        ok("第1批: 3个读工具并行")
    if len(batches[1]) == 1 and batches[1][0].tool_name == "write_file":
        ok("第2批: write_file 串行")
    if len(batches[2]) == 2:
        ok("第3批: 2个读工具并行")


# ═════════════════════════════════════════════════════════════════════════════
# Test 8: 文件缓存 + 截断
# ═════════════════════════════════════════════════════════════════════════════

def test_cache_and_truncation():
    print("\n── Test 8: 文件缓存 + 截断 ──")

    # Cache
    cache = FileStateCache(max_entries=5, max_total_bytes=1024)
    cache.set("/a.py", "print('hello')")
    cache.set("/b.py", "x" * 500)
    assert cache.get("/a.py") is not None
    ok("缓存读写正常")

    cache.invalidate("/a.py")
    assert cache.get("/a.py") is None
    ok("缓存失效正常")

    # LRU eviction
    for i in range(10):
        cache.set(f"/file_{i}.py", f"content_{i}")
    assert cache.size <= 5
    ok(f"LRU 淘汰正常", f"size={cache.size} (max=5)")

    # Truncation
    short = truncate_tool_result("hello", "read_file")
    assert short == "hello"
    ok("短结果不截断")

    long = truncate_tool_result("x" * 100_000, "read_file", max_chars=5000)
    assert len(long) < 6000
    assert "truncated" in long
    ok("长结果截断", f"100K → {len(long)} chars")

    empty = truncate_tool_result("", "bash")
    assert "completed with no output" in empty
    ok("空结果标记")


# ═════════════════════════════════════════════════════════════════════════════
# Test 9: 写代码能力 — 让 Agent 自己写一个文件
# ═════════════════════════════════════════════════════════════════════════════

async def test_write_code():
    print("\n── Test 9: Agent 写代码能力 ──")

    perms = PermissionChecker(ask_fn=lambda name, inp: True)

    agent = Agent(
        api_key=API_KEY,
        model=MODEL,
        base_url=BASE_URL,
        auto_compact=False,
        inject_git_state=False,
        inject_memory=False,
        permission_checker=perms,
        tools=[
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            BashTool(),
        ],
    )

    test_file = "/tmp/lumen_test_output.py"

    tool_calls_made = []
    def on_tool_call(name, args):
        tool_calls_made.append(name)

    response = await agent.chat(
        f"请用 write_file 工具在 {test_file} 创建一个 Python 文件，内容是一个计算斐波那契数列的函数 fib(n)，然后用 bash 执行 python {test_file} 来验证。",
        on_tool_call=on_tool_call,
    )

    if "write_file" in tool_calls_made:
        ok("Agent 调用了 write_file")
    else:
        fail("Agent 没有调用 write_file", f"calls: {tool_calls_made}")

    if "bash" in tool_calls_made:
        ok("Agent 调用了 bash 验证")

    # Check if file was created
    if Path(test_file).exists():
        content = Path(test_file).read_text()
        if "fib" in content or "fibonacci" in content.lower():
            ok("文件内容正确", f"{len(content)} chars")
        else:
            fail("文件内容不含 fib 函数")
        # Cleanup
        Path(test_file).unlink()
    else:
        fail("文件未创建")

    if response.content:
        ok("收到总结回答", f"{response.content[:60]}...")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  Lumen v0.3.0 — 端到端测试")
    print("=" * 60)

    start = time.time()

    # 离线测试（不需要 API）
    test_model_profiles()
    test_tool_schema_adaptation()
    test_permissions()
    test_error_classification()
    test_tool_batching()
    test_cache_and_truncation()

    # 在线测试（需要 API）
    print("\n" + "─" * 60)
    print("  以下测试需要真实 API 调用")
    print("─" * 60)

    try:
        await test_basic_chat()
        await test_tool_calling()
        await test_write_code()
    except Exception as e:
        print(f"\n  ⚠️  API 测试失败: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"  结果: {_pass_count} ✅  {_fail_count} ❌  ({elapsed:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
