"""The reliability layer: retry-with-jitter and ordered cross-provider fallback.

This is intentionally provider-agnostic — it operates on zero-arg callables, so
it has no dependency on any SDK and is trivially testable. The client wires real
provider calls into it.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")

# Status codes that are safe to retry. Anything else (400/401/403/422, ...) is a
# client error — retrying it just burns money and time, so we fail over instead.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter. Defaults are deliberately conservative."""

    max_attempts: int = 3
    base_delay: float = 0.5  # seconds
    max_delay: float = 8.0
    jitter: float = 0.3  # +/- fraction applied to each computed delay

    def backoff(self, attempt: int) -> float:
        raw = min(self.max_delay, self.base_delay * (2 ** attempt))
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread))


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status is not None:
        try:
            return int(status) in RETRYABLE_STATUS
        except (TypeError, ValueError):
            return False
    # Status-less network errors are generally worth one more try.
    return isinstance(exc, (TimeoutError, ConnectionError))


def with_fallback(
    providers: Iterable[Callable[[], T]],
    *,
    policy: Optional[RetryPolicy] = None,
    sleep: Callable[[float], None] = time.sleep,
    is_retryable: Callable[[Exception], bool] = _is_retryable,
) -> T:
    """Try each provider in order; retry retryable failures with backoff+jitter.

    ``providers`` is an ordered list of zero-arg callables, each already bound to
    its model and params. Returns the first success. If every provider is
    exhausted, re-raises the last exception seen.

    This is the *one* retry layer. Do not wrap it around an SDK that is itself
    retrying, or the delays multiply into silent multi-minute hangs.

    ``sleep`` is injectable so tests and benchmarks run instantly.
    """
    policy = policy or RetryPolicy()
    providers = list(providers)
    if not providers:
        raise ValueError("with_fallback requires at least one provider callable")

    last_exc: Optional[Exception] = None
    for provider in providers:
        for attempt in range(policy.max_attempts):
            try:
                return provider()
            except Exception as exc:  # noqa: BLE001 - re-raised below if terminal
                last_exc = exc
                last_attempt = attempt == policy.max_attempts - 1
                if not is_retryable(exc) or last_attempt:
                    break  # give up on this provider, fall over to the next
                sleep(policy.backoff(attempt))

    assert last_exc is not None  # providers was non-empty, so we tried at least once
    raise last_exc
