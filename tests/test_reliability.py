"""Tests for the one piece that must never silently misbehave: the retry layer."""
from __future__ import annotations

import pytest

from justllm.reliability import RetryPolicy, with_fallback

_INSTANT = lambda _: None  # noqa: E731 - no real sleeping in tests


class _Err(Exception):
    def __init__(self, status: int):
        super().__init__(str(status))
        self.status_code = status


def test_retries_then_succeeds():
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Err(503)
        return "ok"

    out = with_fallback([provider], policy=RetryPolicy(max_attempts=3), sleep=_INSTANT)
    assert out == "ok"
    assert calls["n"] == 3


def test_400_is_not_retried_and_falls_over():
    primary_calls = {"n": 0}

    def primary():
        primary_calls["n"] += 1
        raise _Err(400)  # client error — must not be retried

    out = with_fallback([primary, lambda: "backup"], sleep=_INSTANT)
    assert out == "backup"
    assert primary_calls["n"] == 1


def test_exhausts_all_providers_and_raises_last():
    def boom():
        raise _Err(503)

    with pytest.raises(_Err):
        with_fallback([boom], policy=RetryPolicy(max_attempts=2), sleep=_INSTANT)


def test_empty_providers_is_an_error():
    with pytest.raises(ValueError):
        with_fallback([])
