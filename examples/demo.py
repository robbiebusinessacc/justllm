"""A 30-second tour of justllm — nice for a screen recording or asciinema.

    JUSTLLM_MODEL=ollama_chat/llama3.2:1b python examples/demo.py
    # or: GROQ_API_KEY=gsk_... JUSTLLM_MODEL=groq/llama-3.1-8b-instant python examples/demo.py

Record it with:
    asciinema rec demo.cast -c "JUSTLLM_MODEL=... python examples/demo.py"
"""
import json
import os

from justllm import LLM, compress

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


def main() -> None:
    print("justllm — production LLM calls in three lines\n")

    llm = LLM(MODEL)
    print(">>> llm('In one word, the capital of France?')")
    print("   ", llm("In one word, the capital of France?"), "\n")

    # Context compression on a bulky tool result.
    tool = json.dumps({"rows": [{"id": i, "ok": True} for i in range(80)]}, indent=2)
    result = compress(
        [{"role": "tool", "tool_call_id": "c1", "content": tool}], model="gpt-4o"
    )
    print(">>> compress(tool_output)")
    print(
        f"    {result.tokens_before} -> {result.tokens_after} tokens "
        f"({result.pct_saved:.0f}% saved)\n"
    )

    # Streaming.
    print(">>> for chunk in llm.stream('two-line poem about caching'):")
    print("    ", end="")
    for chunk in llm.stream("Write a two-line poem about caching."):
        print(chunk, end="", flush=True)
    print("\n\nAll on by default.  pip install 'justllm[all]'")


if __name__ == "__main__":
    main()
