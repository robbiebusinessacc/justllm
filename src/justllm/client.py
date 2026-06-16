"""The front door: one client, OpenAI-shaped, opinionated defaults on.

Sync and async calls, streaming, structured output, and a tool-calling agent all
ride the same defaults: compression in, native cache optimization, cross-provider
fallback. Routing is opt-in via a `Router`.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence, Type

from .compress import compress as _compress
from .reliability import RetryPolicy, awith_fallback, with_fallback

_VALID_CACHE = {"prompt", "exact", "off"}

_NEEDS_STRUCTURED = (
    "extract() needs instructor + litellm. Install with:\n"
    "    pip install 'justllm[structured]'"
)


class LLM:
    """Production LLM calls in three lines.

    >>> llm = LLM("anthropic/claude-opus-4-8")
    >>> llm("Summarize this contract.")              # doctest: +SKIP

    Fallback chain, knobs, and opt-in routing::

        llm = LLM(chain=["anthropic/claude-opus-4-8", "openai/gpt-5"])
        llm = LLM(router=Router(small="groq/llama-3.1-8b-instant", large="openai/gpt-4o"))
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        chain: Optional[Sequence[str]] = None,
        compress: bool = True,
        cache: str = "prompt",
        retry: Optional[RetryPolicy] = None,
        router: Optional[Any] = None,
    ) -> None:
        if model is None and not chain and router is None:
            raise ValueError("Provide a model, a chain of models, or a router.")
        if cache not in _VALID_CACHE:
            raise ValueError(
                "cache must be 'prompt', 'exact', or 'off'. Semantic caching is "
                "opt-in (and lossy) — it is never the silent default."
            )
        if chain:
            self.chain = list(chain)
        elif model:
            self.chain = [model]
        else:
            self.chain = []
        self.compress = compress
        self.cache = cache
        self.retry = retry or RetryPolicy()
        self.router = router

    # -- internals -------------------------------------------------------------
    def _prepare(self, prompt: str) -> List[dict]:
        messages = [{"role": "user", "content": prompt}]
        if self.compress:
            messages = _compress(messages).messages
        return messages

    def _models(self, prompt: str) -> List[str]:
        if self.router is not None:
            return [self.router.choose(prompt)]
        return self.chain

    # -- sync ------------------------------------------------------------------
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        from .transports import complete

        messages = self._prepare(prompt)

        def caller(model: str):
            return lambda: complete(model, messages, cache=self.cache, **kwargs)

        return with_fallback(
            [caller(m) for m in self._models(prompt)], policy=self.retry
        )

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield text chunks. Streams from the routed/primary model (no mid-stream
        failover — a broken stream can't be silently retried)."""
        from .transports import stream as _stream

        messages = self._prepare(prompt)
        model = self._models(prompt)[0]
        yield from _stream(model, messages, cache=self.cache, **kwargs)

    def extract(self, schema: Type, prompt: str, **kwargs: Any):
        """Structured output: a validated instance of ``schema`` (a Pydantic model)."""
        try:
            import instructor  # noqa: F401
            import litellm  # noqa: F401
        except Exception as exc:
            raise RuntimeError(_NEEDS_STRUCTURED) from exc

        messages = self._prepare(prompt)

        def caller(model: str):
            def call():
                import instructor
                import litellm

                client = instructor.from_litellm(litellm.completion)
                return client.chat.completions.create(
                    model=model, messages=messages, response_model=schema, **kwargs
                )

            return call

        return with_fallback(
            [caller(m) for m in self._models(prompt)], policy=self.retry
        )

    # -- async -----------------------------------------------------------------
    async def acall(self, prompt: str, **kwargs: Any) -> str:
        from .transports import acomplete

        messages = self._prepare(prompt)

        def caller(model: str):
            async def call():
                return await acomplete(model, messages, cache=self.cache, **kwargs)

            return call

        return await awith_fallback(
            [caller(m) for m in self._models(prompt)], policy=self.retry
        )

    async def aextract(self, schema: Type, prompt: str, **kwargs: Any):
        try:
            import instructor  # noqa: F401
            import litellm  # noqa: F401
        except Exception as exc:
            raise RuntimeError(_NEEDS_STRUCTURED) from exc

        messages = self._prepare(prompt)

        def caller(model: str):
            async def call():
                import instructor
                import litellm

                client = instructor.from_litellm(litellm.acompletion)
                return await client.chat.completions.create(
                    model=model, messages=messages, response_model=schema, **kwargs
                )

            return call

        return await awith_fallback(
            [caller(m) for m in self._models(prompt)], policy=self.retry
        )

    # -- agent -----------------------------------------------------------------
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
