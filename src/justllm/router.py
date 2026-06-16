"""Opt-in model routing — deterministic and cheap, no extra LLM call.

Routes by prompt size: short prompts go to a small/cheap model, longer ones to a
larger model. This is a heuristic, not a quality judge — quality-based cascades
(send-to-cheap-then-escalate) are on the roadmap. Routing is never on by default;
you pass a Router explicitly.
"""
from __future__ import annotations


class Router:
    """Length-based router.

    >>> r = Router(small="groq/llama-3.1-8b-instant", large="openai/gpt-4o")
    >>> LLM(router=r)("hi")            # short -> small model        # doctest: +SKIP
    """

    def __init__(self, *, small: str, large: str, max_small_tokens: int = 400) -> None:
        self.small = small
        self.large = large
        self.max_small_tokens = max_small_tokens

    def choose(self, prompt: str) -> str:
        from .compress import count_tokens

        return self.small if count_tokens(prompt) <= self.max_small_tokens else self.large
