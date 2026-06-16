"""Opt-in model routing — deterministic, no hidden LLM calls.

Two strategies, both opt-in (you pass them explicitly; routing is never on by
default):

- ``Router`` — length-based. Short prompts go to a small/cheap model, longer ones
  to a larger model. One call.
- ``Cascade`` — call the cheap model first and escalate to the strong model only
  if the cheap answer looks inadequate (a cheap heuristic, or your own predicate).
  No quality-judge LLM call; the escalation signal is a plain function.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "i cannot",
    "i can't",
    "i'm not sure",
    "im not sure",
    "unable to",
    "as an ai",
)


def _default_escalate(answer: str) -> bool:
    """Escalate if the cheap answer is empty, trivially short, or a refusal."""
    text = (answer or "").strip()
    if len(text) < 2:
        return True
    low = text.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


class Router:
    """Length-based router.

    >>> LLM(router=Router(small="groq/llama-3.1-8b-instant", large="openai/gpt-4o"))
    ... # doctest: +SKIP
    """

    def __init__(self, *, small: str, large: str, max_small_tokens: int = 400) -> None:
        self.small = small
        self.large = large
        self.max_small_tokens = max_small_tokens

    def choose(self, prompt: str) -> str:
        from .compress import count_tokens

        return self.small if count_tokens(prompt) <= self.max_small_tokens else self.large

    def primary(self, prompt: str) -> str:
        return self.choose(prompt)

    def route(self, prompt: str, run: Callable[[str], Any]) -> Any:
        return run(self.choose(prompt))

    async def aroute(self, prompt: str, arun: Callable[[str], Any]) -> Any:
        return await arun(self.choose(prompt))


class Cascade:
    """Cheap-first cascade: try the small model, escalate to the large one only
    when ``escalate_if(answer)`` is true.

    >>> LLM(router=Cascade(small="groq/llama-3.1-8b-instant", large="openai/gpt-4o"))
    ... # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        small: str,
        large: str,
        escalate_if: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.small = small
        self.large = large
        self.escalate_if = escalate_if or _default_escalate

    def primary(self, prompt: str) -> str:
        # Non-call paths (streaming, extract, agent) use the cheap model.
        return self.small

    def route(self, prompt: str, run: Callable[[str], Any]) -> Any:
        answer = run(self.small)
        return run(self.large) if self.escalate_if(answer) else answer

    async def aroute(self, prompt: str, arun: Callable[[str], Any]) -> Any:
        answer = await arun(self.small)
        return await arun(self.large) if self.escalate_if(answer) else answer
