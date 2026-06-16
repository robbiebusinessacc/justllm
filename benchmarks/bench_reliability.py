"""Reliability benchmark — does the fallback layer actually recover failures?

Uses flaky mock providers (no network), so it runs instantly and
deterministically and proves the retry/failover logic in isolation.

    python -m benchmarks.bench_reliability
"""
from __future__ import annotations

from typing import Dict, List

from justllm.reliability import RetryPolicy, with_fallback


class FakeError(Exception):
    def __init__(self, status: int):
        super().__init__(f"status {status}")
        self.status_code = status


def _flaky(fail_times: int, status: int = 503, label: str = "ok"):
    """A provider that fails `fail_times` times (with `status`) then succeeds."""
    state = {"calls": 0}

    def provider():
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise FakeError(status)
        return label

    provider.state = state  # type: ignore[attr-defined]
    return provider


def run() -> List[Dict]:
    no_sleep = lambda _: None  # noqa: E731 - keep the benchmark instant
    policy = RetryPolicy(max_attempts=3)
    scenarios: List[Dict] = []

    # 1. Transient blips on the primary; recovers via retry (no failover needed).
    p = _flaky(fail_times=2, label="primary")
    out = with_fallback([p], policy=policy, sleep=no_sleep)
    scenarios.append(
        {
            "scenario": "primary blips x2 then ok",
            "result": out,
            "primary_calls": p.state["calls"],
        }
    )

    # 2. Primary down hard; fails over to the secondary.
    primary = _flaky(fail_times=99, label="primary")
    secondary = _flaky(fail_times=0, label="secondary")
    out = with_fallback([primary, secondary], policy=policy, sleep=no_sleep)
    scenarios.append(
        {
            "scenario": "primary down -> secondary",
            "result": out,
            "primary_calls": primary.state["calls"],
            "secondary_calls": secondary.state["calls"],
        }
    )

    # 3. Non-retryable 400: must NOT retry, must fail over immediately.
    bad = _flaky(fail_times=99, status=400, label="primary")
    backup = _flaky(fail_times=0, label="backup")
    out = with_fallback([bad, backup], policy=policy, sleep=no_sleep)
    scenarios.append(
        {
            "scenario": "400 not retried -> backup",
            "result": out,
            "primary_calls": bad.state["calls"],  # expected: 1
            "backup_calls": backup.state["calls"],
        }
    )
    return scenarios


if __name__ == "__main__":
    for s in run():
        print(s)
