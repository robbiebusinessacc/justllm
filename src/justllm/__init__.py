"""justllm — production LLM calls in three lines.

Opinionated defaults: cross-provider fallback, native prompt-cache optimization,
and reversible context compression are on by default. The surface stays tiny on
purpose; when you need a dozen knobs, reach for LiteLLM.
"""
from __future__ import annotations

from .agent import Agent
from .client import LLM
from .compress import CompressionResult, compress
from .reliability import RetryPolicy, awith_fallback, with_fallback
from .router import Router

__version__ = "0.2.0"
__all__ = [
    "LLM",
    "Agent",
    "Router",
    "RetryPolicy",
    "with_fallback",
    "awith_fallback",
    "compress",
    "CompressionResult",
]
