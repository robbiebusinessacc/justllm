"""Provider transport over LiteLLM, with Headroom-powered cache optimization.

We don't re-implement provider SDKs or cache logic — LiteLLM handles the wire
protocol, and Headroom's cache optimizer inserts provider-optimal cache_control
breakpoints (multi-breakpoint, 1024-token minimum, prefix stabilization). Both
are optional: install with `pip install 'justllm[litellm]'` /
`'justllm[compression]'`. Cache optimization no-ops gracefully if Headroom or a
provider optimizer isn't available.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterator, List, Optional

from .observability import call_span

# Exact-match cache. Safe ONLY for deterministic calls; opt-in via cache="exact".
_EXACT_CACHE: Dict[str, str] = {}

# Providers Headroom has a cache optimizer for.
_CACHE_PROVIDERS = {"anthropic", "openai", "google"}


def _require_litellm():
    try:
        import litellm
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Provider transport needs LiteLLM. Install it with:\n"
            "    pip install 'justllm[litellm]'"
        ) from exc
    return litellm


def _provider_of(model: str) -> Optional[str]:
    head = model.split("/", 1)[0].lower() if "/" in model else ""
    if head in _CACHE_PROVIDERS:
        return head
    if head in ("gemini", "vertex_ai"):
        return "google"
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or m.startswith(("o1", "o3", "o4")):
        return "openai"
    if "gemini" in m or "google" in m:
        return "google"
    return None


def _model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _optimize_cache(model: str, messages: List[dict]) -> List[dict]:
    """Insert provider-optimal cache breakpoints via Headroom's registry.

    No-op if Headroom isn't installed or the provider has no optimizer.
    """
    provider = _provider_of(model)
    if provider is None:
        return messages
    try:
        import headroom
    except Exception:
        return messages
    try:
        optimizer = headroom.CacheOptimizerRegistry.get(provider)
        ctx = headroom.OptimizationContext(provider=provider, model=_model_name(model))
        result = optimizer.optimize(messages, ctx)
        return getattr(result, "messages", messages) or messages
    except Exception:
        return messages


def _exact_key(model: str, messages: List[dict], kwargs: dict) -> str:
    blob = json.dumps([model, messages, kwargs], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _prepare_messages(model: str, messages: List[dict], cache: str) -> List[dict]:
    return _optimize_cache(model, messages) if cache == "prompt" else messages


def complete(
    model: str, messages: List[dict], *, cache: str = "prompt", **kwargs: Any
) -> str:
    """One real provider call via LiteLLM. Returns the assistant text."""
    litellm = _require_litellm()

    key = None
    if cache == "exact":
        key = _exact_key(model, messages, kwargs)
        cached = _EXACT_CACHE.get(key)
        if cached is not None:
            return cached

    call_messages = _prepare_messages(model, messages, cache)
    with call_span(model, "chat") as rec:
        resp = litellm.completion(model=model, messages=call_messages, **kwargs)
        rec.record(resp)
    text = resp.choices[0].message.content or ""

    if key is not None:
        _EXACT_CACHE[key] = text
    return text


async def acomplete(
    model: str, messages: List[dict], *, cache: str = "prompt", **kwargs: Any
) -> str:
    """Async sibling of `complete`."""
    litellm = _require_litellm()

    key = None
    if cache == "exact":
        key = _exact_key(model, messages, kwargs)
        cached = _EXACT_CACHE.get(key)
        if cached is not None:
            return cached

    call_messages = _prepare_messages(model, messages, cache)
    with call_span(model, "chat") as rec:
        resp = await litellm.acompletion(model=model, messages=call_messages, **kwargs)
        rec.record(resp)
    text = resp.choices[0].message.content or ""

    if key is not None:
        _EXACT_CACHE[key] = text
    return text


def stream(
    model: str, messages: List[dict], *, cache: str = "prompt", **kwargs: Any
) -> Iterator[str]:
    """Yield text chunks from a streaming completion."""
    litellm = _require_litellm()
    call_messages = _prepare_messages(model, messages, cache)
    response = litellm.completion(
        model=model, messages=call_messages, stream=True, **kwargs
    )
    for chunk in response:
        try:
            piece = chunk.choices[0].delta.content
        except Exception:
            piece = None
        if piece:
            yield piece
