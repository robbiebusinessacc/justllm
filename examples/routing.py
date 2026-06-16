"""Routing: send work to the right model, opt-in and deterministic.

Set JUSTLLM_SMALL / JUSTLLM_LARGE to two models you can reach.

    GROQ_API_KEY=gsk_... JUSTLLM_SMALL=groq/llama-3.1-8b-instant \
        JUSTLLM_LARGE=openai/gpt-4o python examples/routing.py
"""
import os

from justllm import LLM, Cascade, Router

SMALL = os.getenv("JUSTLLM_SMALL", "groq/llama-3.1-8b-instant")
LARGE = os.getenv("JUSTLLM_LARGE", "openai/gpt-4o")


def main() -> None:
    # Length-based: short prompts -> small/cheap model, long -> large.
    by_length = LLM(router=Router(small=SMALL, large=LARGE, max_small_tokens=200))
    print("router:  ", by_length("hi"))

    # Cascade: call the cheap model first, escalate only if the answer looks
    # inadequate (default heuristic, or pass your own escalate_if predicate).
    smart = LLM(router=Cascade(small=SMALL, large=LARGE))
    print("cascade: ", smart("Explain the CAP theorem in one sentence."))


if __name__ == "__main__":
    main()
