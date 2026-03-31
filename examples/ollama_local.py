"""
Local model via Ollama (or any OpenAI-compatible local server).

Requires Ollama running: https://ollama.ai
    ollama pull llama3.1
    ollama serve

Run:
    python examples/ollama_local.py
"""

import asyncio
from engram import Agent


async def main() -> None:
    # Ollama exposes an OpenAI-compatible API on port 11434
    agent = Agent(
        api_key="ollama",          # Ollama doesn't need a real key
        model="llama3.1",
        base_url="http://localhost:11434/v1",
        context_window=128_000,    # Override since ollama models vary
        auto_compact=True,
        inject_git_state=True,
    )

    print(f"Local agent: {agent}\n")

    response = await agent.chat(
        "Write a Python function that checks if a string is a palindrome."
    )
    print(f"Agent:\n{response.content}")
    print(f"\nTokens: {response.token_usage.total}/{agent.context_window}")


if __name__ == "__main__":
    asyncio.run(main())
