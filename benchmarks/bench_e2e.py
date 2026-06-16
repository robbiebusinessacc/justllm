"""End-to-end latency benchmark — makes a REAL model call.

Prefers a hosted model if an API key is set, otherwise falls back to a local
Ollama model (free, keyless). Skips cleanly if neither is available, so it never
spends money or fails in CI by surprise.

    OPENAI_API_KEY=sk-... python -m benchmarks.bench_e2e   # hosted
    python -m benchmarks.bench_e2e                          # local Ollama, if running
"""
from __future__ import annotations

import json
import os
import time
import urllib.request


def _ollama_model() -> str | None:
    """First locally-available Ollama model, as a LiteLLM model string."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=2) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
        return f"ollama_chat/{models[0]}" if models else None
    except Exception:
        return None


def _model() -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-haiku-4-5"
    if os.getenv("GROQ_API_KEY"):
        return "groq/llama-3.1-8b-instant"
    return _ollama_model()


def run() -> dict:
    model = _model()
    if model is None:
        print("e2e: SKIPPED (set an API key or run Ollama to enable)")
        return {"skipped": True}

    from justllm import LLM

    llm = LLM(model)
    start = time.perf_counter()
    reply = llm("In one word: the capital of France?")
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"e2e: model={model} latency={latency_ms:.0f}ms reply={reply[:60]!r}")
    return {"skipped": False, "model": model, "latency_ms": latency_ms}


if __name__ == "__main__":
    run()
