"""Async: await a call (and extract) without blocking.

    JUSTLLM_MODEL=ollama_chat/llama3.2:1b python examples/async_calls.py
"""
import asyncio
import os

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


async def main() -> None:
    llm = LLM(MODEL)
    # Fire several calls concurrently.
    questions = [
        "In one word, the capital of Japan?",
        "In one word, the capital of Italy?",
        "In one word, the capital of Spain?",
    ]
    answers = await asyncio.gather(*(llm.acall(q) for q in questions))
    for q, a in zip(questions, answers, strict=True):
        print(f"{q}  ->  {a}")


if __name__ == "__main__":
    asyncio.run(main())
