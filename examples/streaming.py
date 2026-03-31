"""
Streaming response example.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/streaming.py
"""

import asyncio
import os
from engram import Agent


async def main() -> None:
    agent = Agent(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
        system_prompt="You are a concise technical writer.",
    )

    print("User: Write a short explanation of async/await in Python.\n")
    print("Agent: ", end="", flush=True)

    async for chunk in agent.stream("Write a short explanation of async/await in Python."):
        print(chunk, end="", flush=True)

    print(f"\n\n[Total tokens used: {agent.token_usage.total}]")


if __name__ == "__main__":
    asyncio.run(main())
