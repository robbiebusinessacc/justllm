"""Batch: run many prompts at once, bounded by a concurrency limit.

    JUSTLLM_MODEL=ollama_chat/llama3.2:1b python examples/batch.py
"""
import os

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


def main() -> None:
    llm = LLM(MODEL)

    countries = ["France", "Japan", "Brazil", "Egypt"]
    prompts = [f"In one word, the capital of {c}?" for c in countries]

    # Eight in flight at a time; results come back in order.
    answers = llm.map(prompts, concurrency=4)
    for country, answer in zip(countries, answers, strict=True):
        print(f"{country}: {answer.strip()}")

    # Embeddings work the same way (needs an embedding-capable provider):
    # vectors = llm.embed(["hello", "world"], model="openai/text-embedding-3-small")


if __name__ == "__main__":
    main()
