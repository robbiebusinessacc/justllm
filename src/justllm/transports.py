"""Provider transport.

A thin layer over LiteLLM. We do not re-implement provider SDKs — justllm's
value is the opinionated default layer (fallback, caching, compression), not the
wire protocol. LiteLLM is an optional dependency:

    pip install 'justllm[litellm]'
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

# Tiny in-process exact-match cache. Safe ONLY for deterministic calls
# (temperature 0). Never used unless the caller explicitly asks for cache="exact".
_EXACT_CACHE: Dict[str, str] = {}


def _require_litellm():
    try:
        import litellm
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Provider transport needs LiteLLM. Install it with:\n"
            "    pip install 'justllm[litellm]'"
        ) from exc
    return litellm


def _is_anthropic(model: str) -> bool:
    m = model.lower()
    return "anthropic" in m or "claude" in m


def _exact_key(model: str, messages: List[dict], kwargs: dict) -> str:
    blob = json.dumps([model, messages, kwargs], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _with_anthropic_cache(messages: List[dict]) -> List[dict]:
    """Add a single Anthropic ``cache_control`` breakpoint at the first
    system/user text block. OpenAI caches automatically, so this is skipped
    there. LiteLLM passes the breakpoint through to the Anthropic API.
    """
    out: List[dict] = []
    marked = False
    for msg in messages:
        content = msg.get("content")
        if (
            not marked
            and msg.get("role") in ("system", "user")
            and isinstance(content, str)
            and content
        ):
            out.append(
                {
                    **msg,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
            marked = True
        else:
            out.append(msg)
    return out


def complete(
    model: str, messages: List[dict], *, cache: str = "prompt", **kwargs: Any
) -> str:
    """One real provider call via LiteLLM. Returns the assistant text.

    ``cache``: "prompt" applies native prompt caching (Anthropic breakpoint /
    OpenAI automatic); "exact" adds a local exact-match cache for deterministic
    calls; "off" disables both.
    """
    litellm = _require_litellm()

    key = None
    if cache == "exact":
        key = _exact_key(model, messages, kwargs)
        cached = _EXACT_CACHE.get(key)
        if cached is not None:
            return cached

    call_messages = messages
    if cache == "prompt" and _is_anthropic(model):
        call_messages = _with_anthropic_cache(messages)

    resp = litellm.completion(model=model, messages=call_messages, **kwargs)
    text = resp.choices[0].message.content or ""

    if key is not None:
        _EXACT_CACHE[key] = text
    return text
