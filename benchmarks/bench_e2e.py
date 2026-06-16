"""End-to-end benchmark — makes REAL model calls.

Prefers a hosted model if an API key is set (OpenAI / Anthropic / Groq),
otherwise falls back to a local Ollama model (free, keyless). Skips cleanly if
neither is available, so it never spends money or fails in CI by surprise.

Exercises the full call path: a basic completion, structured output, and an
agent tool call. The structured-output and agent probes need a reasonably
capable model — tiny local models will report WEAK MODEL rather than crash.

    GROQ_API_KEY=gsk_... python -m benchmarks.bench_e2e   # free, capable
    python -m benchmarks.bench_e2e                         # local Ollama, if running
"""
from __future__ import annotations

import json
import os
import time
import urllib.request


def _ollama_model() -> str | None:
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


def _probe_extract(model: str, results: dict) -> None:
    try:
        from pydantic import BaseModel

        from justllm import LLM

        class _City(BaseModel):
            name: str
            country: str

        city = LLM(model, compress=False).extract(
            _City, "The Eiffel Tower is in Paris, France."
        )
        ok = isinstance(city, _City) and bool(city.name)
        print(f"e2e: extract -> {city!r} [{'OK' if ok else 'WEAK MODEL'}]")
        results["extract_ok"] = ok
    except Exception as exc:
        print(f"e2e: extract -> FAILED ({type(exc).__name__}) [weak model / no support]")
        results["extract_ok"] = False


def _probe_agent(model: str, results: dict) -> None:
    try:
        from justllm import LLM

        agent = LLM(model, compress=False).agent(
            system="Use the add tool for any arithmetic.", max_steps=4
        )
        seen: dict = {}

        @agent.tool
        def add(a: int, b: int) -> int:
            "Add two integers."
            seen["args"] = (a, b)
            return a + b

        out = agent.run("What is 21 + 21? Use the add tool.")
        called = seen.get("args") is not None
        print(f"e2e: agent   -> {out[:50]!r} tool_called={called} "
              f"[{'OK' if called else 'WEAK MODEL'}]")
        results["agent_tool_called"] = called
    except Exception as exc:
        print(f"e2e: agent   -> FAILED ({type(exc).__name__}) [weak model / no tools]")
        results["agent_tool_called"] = False


def run() -> dict:
    model = _model()
    if model is None:
        print("e2e: SKIPPED (set an API key or run Ollama to enable)")
        return {"skipped": True}

    from justllm import LLM

    start = time.perf_counter()
    reply = LLM(model)("In one word: the capital of France?")
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"e2e: model={model} latency={latency_ms:.0f}ms reply={reply[:60]!r}")

    results = {"skipped": False, "model": model, "latency_ms": latency_ms}
    _probe_extract(model, results)
    _probe_agent(model, results)
    return results


if __name__ == "__main__":
    run()
