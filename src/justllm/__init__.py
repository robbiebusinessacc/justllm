"""justllm — production LLM calls in three lines.

Opinionated defaults: cross-provider fallback, native prompt caching, and
reversible context compression are on by default. The surface stays tiny on
purpose; when you need a dozen knobs, reach for LiteLLM.
"""
from __future__ import annotations

from .agent import Agent
from .client import LLM
from .compress import CompressionResult, compress
from .reliability import RetryPolicy, with_fallback

__version__ = "0.1.0"
__all__ = [
    "LLM",
    "Agent",
    "RetryPolicy",
    "with_fallback",
    "compress",
    "CompressionResult",
]
