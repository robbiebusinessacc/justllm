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

import math
from typing import Any, Callable, Optional, Sequence

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


# -- embedding-based escalation -----------------------------------------------

_DEFAULT_UNCERTAINTY_EXEMPLARS = (
    "I don't know the answer to that.",
    "I'm not sure about this.",
    "I cannot determine that from the given information.",
    "It's unclear and I can't say for certain.",
    "I'm unable to help with that request.",
    "There isn't enough information to answer.",
)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _fastembed_embedder() -> Callable[[Sequence[str]], list]:
    from fastembed import TextEmbedding

    model = TextEmbedding()  # bge-small, local ONNX (~30MB, downloaded once)

    def embed(texts: Sequence[str]) -> list:
        return [list(v) for v in model.embed(list(texts))]

    return embed


def _default_embedder() -> Callable[[Sequence[str]], list]:
    try:
        return _fastembed_embedder()
    except Exception as exc:
        raise RuntimeError(
            "embedding_escalator needs an embedder. Install one with "
            "`pip install 'justllm[embeddings]'`, or pass embed=your_own_fn."
        ) from exc


def embedding_escalator(
    *,
    embed: Optional[Callable[[Sequence[str]], list]] = None,
    exemplars: Optional[Sequence[str]] = None,
    threshold: float = 0.65,
) -> Callable[[str], bool]:
    """An escalation predicate for ``Cascade`` based on *semantic* uncertainty.

    Embeds the cheap model's answer and escalates when it's semantically close to
    a refusal / hedge / uncertainty exemplar — catching paraphrases the keyword
    heuristic misses ("it's hard to say", "there isn't enough information"). It
    detects *hedging*, not factual errors.

    ``embed(texts) -> list[vector]`` defaults to fastembed (bge-small, local ONNX;
    ships with the ``[embeddings]`` / ``[compression]`` extras). Raise ``threshold``
    to escalate less. Use it as::

        Cascade(small=cheap, large=big, escalate_if=embedding_escalator())
    """
    embed_fn = embed or _default_embedder()
    exemplar_texts = list(exemplars) if exemplars else list(_DEFAULT_UNCERTAINTY_EXEMPLARS)
    exemplar_vecs = embed_fn(exemplar_texts)

    def escalate_if(answer: str) -> bool:
        text = (answer or "").strip()
        if len(text) < 2:
            return True
        vec = embed_fn([text])[0]
        return max(_cosine(vec, ev) for ev in exemplar_vecs) >= threshold

    return escalate_if
