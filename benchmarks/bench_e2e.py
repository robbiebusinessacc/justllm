"""End-to-end latency benchmark — makes a REAL provider call.

Skips automatically unless an API key is present, so it never spends money or
fails in CI by surprise. Run it deliberately:

    OPENAI_API_KEY=sk-... python -m benchmarks.bench_e2e
"""
from __future__ import annotations

import os
import time


def _model() -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-haiku-4-5"
    return None


def run() -> dict:
    model = _model()
    if model is None:
        print("e2e: SKIPPED (set OPENAI_API_KEY or ANTHROPIC_API_KEY to run)")
        return {"skipped": True}

    from justllm import LLM

    llm = LLM(model)
    start = time.perf_counter()
    reply = llm("Reply with exactly: ok")
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"e2e: model={model} latency={latency_ms:.0f}ms reply={reply[:40]!r}")
    return {"skipped": False, "model": model, "latency_ms": latency_ms}


if __name__ == "__main__":
    run()
