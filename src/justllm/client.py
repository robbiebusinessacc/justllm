"""The front door: one client, OpenAI-shaped, opinionated defaults on.

Sync and async calls, streaming, structured output, and a tool-calling agent all
ride the same defaults: compression in, native cache optimization, cross-provider
fallback. Routing is opt-in via a `Router`.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Sequence, Type

from .compress import compress as _compress
from .reliability import RetryPolicy, awith_fallback, with_fallback

_VALID_CACHE = {"prompt", "exact", "off"}

# Sensible default embedding model per provider; pass model= for anything else.
_DEFAULT_EMBED_MODELS = {
    "openai": "text-embedding-3-small",
    "google": "text-embedding-004",
}

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

    def _primary_model(self, prompt: str) -> str:
        if self.router is not None:
            return self.router.primary(prompt)
        return self.chain[0]

    def _chain_for(self, prompt: str) -> List[str]:
        return [self._primary_model(prompt)] if self.router is not None else self.chain

    def _reply(self, messages: List[dict], route_key: str, **kwargs: Any) -> str:
        """The shared sync engine: compress, then route or fall back over a chain.

        Takes a full message list (so single calls and multi-turn chat share it)
        and a route_key (the latest user text) for routing decisions.
        """
        from .transports import complete

        if self.compress:
            messages = _compress(messages).messages

        def run(model: str) -> str:
            return complete(model, messages, cache=self.cache, **kwargs)

        if self.router is not None:
            def routed(model: str) -> str:
                return with_fallback([lambda: run(model)], policy=self.retry)

            return self.router.route(route_key, routed)

        def caller(model: str) -> Callable[[], str]:
            return lambda: run(model)

        return with_fallback([caller(m) for m in self.chain], policy=self.retry)

    # -- sync ------------------------------------------------------------------
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self._reply([{"role": "user", "content": prompt}], prompt, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Yield text chunks. Streams from the routed/primary model (no mid-stream
        failover — a broken stream can't be silently retried, and a cascade can't
        judge an answer it hasn't finished receiving)."""
        from .transports import stream as _stream

        messages = self._prepare(prompt)
        model = self._primary_model(prompt)
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
            [caller(m) for m in self._chain_for(prompt)], policy=self.retry
        )

    # -- async -----------------------------------------------------------------
    async def _areply(self, messages: List[dict], route_key: str, **kwargs: Any) -> str:
        from .transports import acomplete

        if self.compress:
            messages = _compress(messages).messages

        def caller(model: str):
            async def call():
                return await acomplete(model, messages, cache=self.cache, **kwargs)

            return call

        if self.router is not None:
            async def arun(model: str):
                return await awith_fallback([caller(model)], policy=self.retry)

            return await self.router.aroute(route_key, arun)

        return await awith_fallback([caller(m) for m in self.chain], policy=self.retry)

    async def acall(self, prompt: str, **kwargs: Any) -> str:
        return await self._areply([{"role": "user", "content": prompt}], prompt, **kwargs)

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
            [caller(m) for m in self._chain_for(prompt)], policy=self.retry
        )

    # -- batch -----------------------------------------------------------------
    async def amap(
        self, prompts: Sequence[str], *, concurrency: int = 8, **kwargs: Any
    ) -> List[str]:
        """Run many prompts concurrently, bounded by `concurrency`, in order."""
        import asyncio

        sem = asyncio.Semaphore(concurrency)

        async def one(prompt: str) -> str:
            async with sem:
                return await self.acall(prompt, **kwargs)

        return await asyncio.gather(*(one(p) for p in prompts))

    def map(
        self, prompts: Sequence[str], *, concurrency: int = 8, **kwargs: Any
    ) -> List[str]:
        """Sync wrapper over `amap`. Inside an event loop, await `amap` instead."""
        import asyncio

        return asyncio.run(self.amap(prompts, concurrency=concurrency, **kwargs))

    # -- embeddings ------------------------------------------------------------
    def embed(self, texts: "str | Sequence[str]", *, model: Optional[str] = None):
        """Embed text. Returns one vector for a string, a list of vectors for a list.

        Uses an embedding model, not the chat model. Defaults to a sensible one
        for OpenAI/Google; pass `model=` for any other provider.
        """
        from .transports import _provider_of, _require_litellm

        litellm = _require_litellm()
        provider = _provider_of(self.chain[0]) if self.chain else None
        emb_model = model or (_DEFAULT_EMBED_MODELS.get(provider) if provider else None)
        if emb_model is None:
            raise ValueError(
                "Specify an embedding model, e.g. "
                "llm.embed(texts, model='openai/text-embedding-3-small')."
            )
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        resp = litellm.embedding(model=emb_model, input=inputs)
        vectors = [item["embedding"] for item in resp.data]
        return vectors[0] if single else vectors

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

    def chat(self, *, system: Optional[str] = None):
        """A multi-turn conversation that remembers history across turns."""
        from .chat import Chat

        return Chat(self, system=system)

    # -- evaluation ------------------------------------------------------------
    def judge(self, output: str, *, criteria: str, reference: Optional[str] = None,
              scale: int = 5, **kwargs: Any):
        """LLM-as-judge: grade one output against a criterion. Returns a Verdict."""
        from .eval import judge as _judge

        return _judge(self, output, criteria=criteria, reference=reference,
                      scale=scale, **kwargs)

    def evaluate(self, cases: Any, *, scorer: Optional[Any] = None, grader: Any = None,
                 concurrency: int = 8, **kwargs: Any):
        """Run and score a set of cases. Returns an EvalReport (pass rate + per-case)."""
        from .eval import evaluate as _evaluate

        return _evaluate(self, cases, scorer=scorer, grader=grader,
                         concurrency=concurrency, **kwargs)
