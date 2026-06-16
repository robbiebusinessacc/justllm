"""justllm — production LLM calls in three lines.

Opinionated defaults: cross-provider fallback, native prompt-cache optimization,
and reversible context compression are on by default. The surface stays tiny on
purpose; when you need a dozen knobs, reach for LiteLLM.
"""
from __future__ import annotations

from . import prompts
from .agent import Agent
from .chat import Chat
from .client import LLM
from .compress import CompressionResult, compress
from .reliability import RetryPolicy, awith_fallback, with_fallback
from .router import Cascade, Router, embedding_escalator

__version__ = "0.7.0"
__all__ = [
    "LLM",
    "Agent",
    "Chat",
    "Router",
    "Cascade",
    "embedding_escalator",
    "RetryPolicy",
    "with_fallback",
    "awith_fallback",
    "compress",
    "CompressionResult",
    "prompts",
]
