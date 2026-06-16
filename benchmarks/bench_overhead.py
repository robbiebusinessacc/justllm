"""Overhead benchmark — how much wall-clock does the compression pass add?

The justllm value prop only holds if the layer is thin, so this times the
compression pass per fixture. End-to-end call overhead lands once the provider
transport is wired.

    python -m benchmarks.bench_overhead
"""
from __future__ import annotations

import time
from typing import Dict, List

from justllm.compress import compress

from .fixtures import all_fixtures


def run(repeats: int = 20) -> List[Dict]:
    rows = []
    for name, payload in all_fixtures().items():
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": payload}]
        compress(messages)  # warmup (tokenizer load, import, etc.)
        start = time.perf_counter()
        for _ in range(repeats):
            compress(messages)
        elapsed = (time.perf_counter() - start) / repeats
        rows.append({"fixture": name, "ms_per_call": elapsed * 1000})
    return rows


if __name__ == "__main__":
    for r in run():
        print(f"{r['fixture']:<22}{r['ms_per_call']:>8.2f} ms")
