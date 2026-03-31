"""
Basic usage — multi-turn chat with automatic context management.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/basic.py
"""

import asyncio
import os
from engram import Agent


async def main() -> None:
    # ── Create the agent — that's all you need ────────────────────────────────
    agent = Agent(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
        # Everything below is optional and has sensible defaults:
        # system_prompt="You are a Python expert.",
        # inject_git_state=True,   # inject git branch/status into system prompt
        # inject_memory=True,      # auto-load ENGRAM.md memory files
        # auto_compact=True,       # auto-compress context when approaching limit
    )
    print(f"Agent ready: {agent}\n")

    # ── Multi-turn conversation ───────────────────────────────────────────────
    questions = [
        "What is the difference between a list and a tuple in Python?",
        "When should I prefer one over the other?",
        "Give me a real-world example where a tuple is the better choice.",
    ]

    for q in questions:
        print(f"User: {q}")
        response = await agent.chat(q)
        print(f"Agent: {response.content}")
        print(f"  [tokens: {response.token_usage.total}/{agent.context_window}]\n")

    # ── Save the session for later ────────────────────────────────────────────
    agent.save_session("/tmp/engram_session.json")
    print("Session saved to /tmp/engram_session.json")


if __name__ == "__main__":
    asyncio.run(main())
