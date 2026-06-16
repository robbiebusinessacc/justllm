"""Basics: a call, a cross-provider fallback chain, and caching.

    JUSTLLM_MODEL=ollama_chat/llama3.2:1b python examples/basics.py
"""
import os

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


def main() -> None:
    # 1. the three-line version
    llm = LLM(MODEL)
    print("call:    ", llm("In one word, the capital of France?"))

    # 2. cross-provider fallback: try the first model, fall back on failure.
    robust = LLM(chain=[MODEL, "groq/llama-3.1-8b-instant"])
    print("fallback:", robust("Say hello in one word."))

    # 3. caching: native prompt-cache optimization is on by default; add an
    #    exact-match cache for identical, deterministic calls.
    cached = LLM(MODEL, cache="exact")
    cached("2 + 2 = ?", temperature=0)
    print("cached:  ", cached("2 + 2 = ?", temperature=0))  # served from cache


if __name__ == "__main__":
    main()
