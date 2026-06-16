"""Streaming: print tokens as they arrive.

    JUSTLLM_MODEL=ollama_chat/llama3.2:1b python examples/streaming.py
"""
import os

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


def main() -> None:
    llm = LLM(MODEL)
    for chunk in llm.stream("Write a two-line poem about caching."):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
