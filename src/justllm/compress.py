"""Context compression: a thin adapter over Headroom.

Headroom (https://github.com/chopratejas/headroom) does content-aware,
reversible compression of the dynamic junk that bloats agent calls — tool
outputs, logs, JSON, RAG dumps. We pin its call behind one adapter so a
signature change upstream is a one-line fix here, and we degrade to a
conservative structural pass when Headroom is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class CompressionResult:
    messages: List[dict]
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def pct_saved(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 100.0 * self.tokens_saved / self.tokens_before


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Approximate token count via tiktoken; falls back to a chars/4 heuristic.

    The heuristic keeps the benchmark runnable without tiktoken installed; it is
    only an estimate, so install tiktoken for real numbers.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _messages_text(messages: List[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def compress(
    messages: List[dict], model: str = "gpt-4o", config: Any = None
) -> CompressionResult:
    """Compress dynamic context before it reaches the model.

    Tries Headroom first. If Headroom is unavailable, applies a conservative
    structural fallback so the call still works — but Headroom is the intended
    engine and the one the benchmarks are meant to measure.

    `config` is an optional Headroom ``CompressConfig`` to tune what gets
    compressed (e.g. protect recent turns, target ratio, compress user messages).
    """
    before = count_tokens(_messages_text(messages), model)
    out = _headroom_compress(messages, model, config)
    if out is None:
        out = _naive_compress(messages)
    after = count_tokens(_messages_text(out), model)
    return CompressionResult(messages=out, tokens_before=before, tokens_after=after)


def _headroom_compress(
    messages: List[dict], model: str, config: Any = None
) -> Optional[List[dict]]:
    try:
        import headroom  # type: ignore
    except Exception:
        return None
    try:
        # Documented Python API: compress(messages, model=..., config=...).
        result = headroom.compress(messages, model=model, config=config)
    except Exception:
        return None
    # Be liberal in what we accept back across Headroom versions:
    # current API returns a CompressResult object exposing `.messages`; older or
    # alternate shapes may hand back a dict or a bare list.
    msgs = getattr(result, "messages", None)
    if msgs is not None:
        return msgs
    if isinstance(result, dict) and "messages" in result:
        return result["messages"]
    if isinstance(result, list):
        return result
    return None


def _naive_compress(messages: List[dict]) -> List[dict]:
    """Last-resort baseline: collapse redundant whitespace.

    This exists only so the pipeline degrades gracefully without Headroom. It is
    NOT the product — real savings come from content-aware compression.
    """
    import re

    out: List[dict] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            content = re.sub(r"[ \t]+", " ", content)
            content = re.sub(r"\n{3,}", "\n\n", content)
        out.append({**m, "content": content})
    return out
