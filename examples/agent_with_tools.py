"""A tool-calling agent: register Python functions, let the model use them.

Needs a tool-capable model and `pip install 'justllm[litellm]'`.

    GROQ_API_KEY=gsk_... JUSTLLM_MODEL=groq/llama-3.1-8b-instant \
        python examples/agent_with_tools.py
"""
import os

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


def main() -> None:
    agent = LLM(MODEL).agent(system="Use the tools for math and weather.", max_steps=6)

    @agent.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @agent.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"{city}: sunny, 72F"

    print(agent.run("What is 19 + 23, and what's the weather in Boston?"))


if __name__ == "__main__":
    main()
