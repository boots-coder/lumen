"""
Memory files (ENGRAM.md) + session persistence + model switching.

This demo shows:
  1. How ENGRAM.md files are auto-discovered and injected
  2. How to save and restore a session across process restarts
  3. How to use a different model for compaction (e.g. cheap model for chat,
     strong model for summarisation)

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/with_memory.py
"""

import asyncio
import os
from pathlib import Path
from engram import Agent

SESSION_FILE = "/tmp/engram_demo_session.json"


async def first_session() -> None:
    """Simulate a first coding session."""
    print("=== First session ===\n")

    # Point the agent at a project directory that contains an ENGRAM.md
    # In a real project, ENGRAM.md would have team conventions, architecture
    # notes, etc. Engram auto-discovers it and injects it into every call.
    project_dir = Path(__file__).parent.parent  # engram repo root

    agent = Agent(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
        cwd=project_dir,
        inject_git_state=True,   # adds branch/status/last-5-commits
        inject_memory=True,      # loads ENGRAM.md files
        auto_compact=True,
    )

    r1 = await agent.chat("What files are in a typical Python SDK project?")
    print(f"Agent: {r1.content[:300]}...\n")

    r2 = await agent.chat("What should go in the pyproject.toml for such a project?")
    print(f"Agent: {r2.content[:300]}...\n")

    print(f"Tokens used: {agent.token_usage.total}/{agent.context_window}")

    # Save session — the full conversation is preserved
    agent.save_session(SESSION_FILE)
    print(f"\nSession saved → {SESSION_FILE}")


async def second_session() -> None:
    """Simulate resuming the session later (different process, same memory)."""
    print("\n=== Second session (resumed) ===\n")

    agent = Agent.load_session(
        SESSION_FILE,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )

    print(f"Restored: {agent}")
    print(f"Message history: {len(agent.messages)} messages\n")

    # The agent remembers everything from the first session
    r = await agent.chat(
        "Based on our earlier discussion, what's the most important thing "
        "to get right in a Python SDK's public API?"
    )
    print(f"Agent: {r.content}\n")


async def multi_model_example() -> None:
    """Use a fast model for chat, a stronger model for compaction."""
    print("\n=== Multi-model compaction ===\n")

    agent = Agent(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="gpt-4o-mini",           # cheap model for day-to-day chat
        compact_model="gpt-4o",        # strong model for compaction summaries
        compact_api_key=os.environ.get("OPENAI_API_KEY", ""),
        auto_compact=True,
    )
    print(f"Agent: {agent}")
    print("Chat model: gpt-4o-mini | Compact model: gpt-4o")
    print("(Compaction runs automatically when context fills up)\n")

    r = await agent.chat("Tell me about the CAP theorem in distributed systems.")
    print(f"Agent: {r.content[:200]}...")


async def main() -> None:
    await first_session()
    await second_session()

    if os.environ.get("OPENAI_API_KEY"):
        await multi_model_example()


if __name__ == "__main__":
    asyncio.run(main())
