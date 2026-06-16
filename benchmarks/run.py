"""Run the full justllm benchmark suite.

    pip install -e '.[benchmarks]'
    python -m benchmarks.run
"""
from __future__ import annotations

from . import bench_e2e, bench_overhead, bench_reliability, bench_savings


def main() -> None:
    bar = "=" * 64
    print(bar)
    print("justllm benchmark suite")
    print(bar)

    # 1. Headline: token + cost savings from compression.
    print(bench_savings.format_table(bench_savings.run()))

    # 2. The layer must be thin — measure compression overhead per call.
    print("\nCompression overhead (per call)")
    print("-" * 64)
    for r in bench_overhead.run():
        print(f"{r['fixture']:<22}{r['ms_per_call']:>8.2f} ms")

    # 3. Fallback must actually recover provider failures.
    print("\nReliability scenarios")
    print("-" * 64)
    for s in bench_reliability.run():
        verdict = "PASS" if s.get("result") else "FAIL"
        calls = {k: v for k, v in s.items() if k.endswith("_calls")}
        print(f"[{verdict}] {s['scenario']:<30} -> {s['result']!r:<12} {calls}")

    # 4. End-to-end latency (real call; skips without an API key).
    print("\nEnd-to-end")
    print("-" * 64)
    bench_e2e.run()


if __name__ == "__main__":
    main()
