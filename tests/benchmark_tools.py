"""
轻量级 Tool Calling Benchmark — 验证多模型格式适配。

每个模型只跑 3 次调用，总成本极低 (~$0.01-0.05)。
测的是：模型能不能正确理解工具 schema 并返回合法的 tool call。

Usage:
    python benchmark_tools.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lumen import Agent, PermissionChecker
from lumen.providers.model_profiles import detect_profile
from lumen.tools import FileReadTool, GlobTool, BashTool, TreeTool

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("GETGOAPI_API_KEY", "sk-your-key-here")
BASE_URL = "https://api.getgoapi.com/v1"

# 每个模型 3 个 prompt，要求必须调工具才能回答
PROMPTS = [
    "用 tree 工具查看当前目录，告诉我顶层有几个文件夹。只回答数字。",
    "用 glob 工具搜索 *.toml 文件，列出找到的文件名。",
    "用 bash 执行 echo hello_lumen，告诉我输出是什么。",
]

# GetGoAPI 支持的模型（cost 从低到高选几个代表）
MODELS = [
    "gpt-4.1-nano-2025-04-14",     # OpenAI 家族 — 最便宜
    "gpt-4.1-mini-2025-04-14",     # OpenAI 家族 — 中等
    "deepseek-chat",                # DeepSeek 家族
]


@dataclass
class BenchResult:
    model: str
    family: str
    prompt: str
    success: bool          # 模型是否产生了合法 tool call
    tool_called: str       # 调用了哪个工具
    latency_ms: float
    error: str = ""


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_one(model: str, prompt: str) -> BenchResult:
    """单次测试：给模型一个 prompt，看它能不能正确调用工具。"""
    profile = detect_profile(model)
    perms = PermissionChecker(ask_fn=lambda n, i: True)

    agent = Agent(
        api_key=API_KEY,
        model=model,
        base_url=BASE_URL,
        auto_compact=False,
        inject_git_state=False,
        inject_memory=False,
        permission_checker=perms,
        tools=[TreeTool(), FileReadTool(), GlobTool(), BashTool()],
    )

    tool_called = ""
    def on_tc(name, args):
        nonlocal tool_called
        tool_called = name

    start = time.monotonic()
    try:
        resp = await agent.chat(prompt, on_tool_call=on_tc)
        latency = (time.monotonic() - start) * 1000

        return BenchResult(
            model=model,
            family=profile.family,
            prompt=prompt[:40],
            success=bool(tool_called),
            tool_called=tool_called or "(none)",
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return BenchResult(
            model=model,
            family=profile.family,
            prompt=prompt[:40],
            success=False,
            tool_called="(error)",
            latency_ms=latency,
            error=str(e)[:80],
        )


async def main():
    print("=" * 70)
    print("  Lumen Tool Calling Benchmark — 轻量版")
    print("  每模型 3 次调用，验证工具格式适配")
    print("=" * 70)

    # Show profiles
    print("\n  模型 Profile 检测:")
    for m in MODELS:
        p = detect_profile(m)
        cap = p.capabilities
        print(f"    {m}")
        print(f"      family={p.family}  schema={p.tool_schema_format.value}")
        print(f"      native_tokens={cap.native_tool_tokens}  parallel={cap.supports_parallel_tool_calls}")

    print(f"\n  共 {len(MODELS)} 模型 × {len(PROMPTS)} prompt = {len(MODELS)*len(PROMPTS)} 次调用\n")

    results: list[BenchResult] = []

    for model in MODELS:
        print(f"  ── {model} ──")
        for i, prompt in enumerate(PROMPTS):
            r = await run_one(model, prompt)
            results.append(r)
            icon = "✅" if r.success else "❌"
            err = f"  [{r.error}]" if r.error else ""
            print(f"    {icon} #{i+1} → {r.tool_called:<12} {r.latency_ms:>6.0f}ms{err}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  {'Model':<35} {'Family':<12} {'Pass':>4} {'Fail':>4} {'Rate':>6} {'Avg ms':>8}")
    print("  " + "─" * 67)

    for model in MODELS:
        model_results = [r for r in results if r.model == model]
        passed = sum(1 for r in model_results if r.success)
        failed = len(model_results) - passed
        rate = passed / len(model_results) * 100
        avg_ms = sum(r.latency_ms for r in model_results) / len(model_results)
        family = model_results[0].family if model_results else "?"
        icon = "✅" if rate == 100 else ("⚠️" if rate >= 50 else "❌")
        print(f"  {icon} {model:<33} {family:<12} {passed:>4} {failed:>4} {rate:>5.0f}% {avg_ms:>7.0f}")

    total_pass = sum(1 for r in results if r.success)
    total = len(results)
    print(f"\n  总计: {total_pass}/{total} ({total_pass/total*100:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
