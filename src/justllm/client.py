"""The front door: one client, OpenAI-shaped, opinionated defaults on.

`__call__` and `extract()` make real provider calls through LiteLLM (an optional
dependency). Compression runs on the way in; cross-provider fallback wraps every
call; native prompt caching is applied per provider.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Type

from .compress import compress as _compress
from .reliability import RetryPolicy, with_fallback

_VALID_CACHE = {"prompt", "exact", "off"}


class LLM:
    """Production LLM calls in three lines.

    Example
    -------
    >>> llm = LLM("anthropic/claude-opus-4-8")
    >>> llm("Summarize this contract.")              # doctest: +SKIP

    With a fallback chain and explicit knobs::

        llm = LLM(
            chain=["anthropic/claude-opus-4-8", "openai/gpt-5"],
            compress=True,     # reversible, dynamic-context only
            cache="prompt",    # never silently means semantic
        )
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        chain: Optional[Sequence[str]] = None,
        compress: bool = True,
        cache: str = "prompt",
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        if model is None and not chain:
            raise ValueError("Provide a model or a chain of models.")
        if cache not in _VALID_CACHE:
            raise ValueError(
                "cache must be 'prompt', 'exact', or 'off'. Semantic caching is "
                "opt-in (and lossy) — it is never the silent default."
            )
        self.chain = list(chain) if chain else [model]  # type: ignore[list-item]
        self.compress = compress
        self.cache = cache
        self.retry = retry or RetryPolicy()

    def _prepare(self, prompt: str) -> List[dict]:
        messages = [{"role": "user", "content": prompt}]
        if self.compress:
            messages = _compress(messages).messages
        return messages

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        from .transports import complete

        messages = self._prepare(prompt)

        def caller(model: str):
            return lambda: complete(model, messages, cache=self.cache, **kwargs)

        return with_fallback(
            [caller(model) for model in self.chain], policy=self.retry
        )

    def extract(self, schema: Type, prompt: str, **kwargs: Any):
        """Structured output: a validated instance of ``schema`` (a Pydantic model).

        Uses instructor over LiteLLM (native structured outputs where available,
        tool-calling fallback otherwise), wrapped in the same fallback chain.
        """
        try:
            import instructor  # noqa: F401
            import litellm  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "extract() needs instructor + litellm. Install with:\n"
                "    pip install 'justllm[structured]'"
            ) from exc

        messages = self._prepare(prompt)

        def caller(model: str):
            def call():
                import instructor
                import litellm

                client = instructor.from_litellm(litellm.completion)
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_model=schema,
                    **kwargs,
                )

            return call

        return with_fallback(
            [caller(model) for model in self.chain], policy=self.retry
        )

    def agent(
        self,
        *,
        system: Optional[str] = None,
        max_steps: int = 8,
        tools: Optional[List[Any]] = None,
    ):
        """A minimal tool-calling loop with a hard step cap."""
        from .agent import Agent

        return Agent(self, system=system, max_steps=max_steps, tools=tools)
