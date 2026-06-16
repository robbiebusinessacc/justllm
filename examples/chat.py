"""Chat: a multi-turn conversation that remembers what came before.

    JUSTLLM_MODEL=ollama_chat/llama3.2:1b python examples/chat.py
"""
import os

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


def main() -> None:
    chat = LLM(MODEL).chat(system="You are concise.")

    print("you: Who wrote Dune?")
    print("bot:", chat.send("Who wrote Dune?").strip())

    print("\nyou: And who wrote the sequel?")
    # No need to repeat context — the chat remembers the first turn.
    print("bot:", chat.send("And who wrote the sequel?").strip())

    print(f"\n({len(chat.messages)} messages in history)")


if __name__ == "__main__":
    main()
