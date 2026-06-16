"""Optional OpenTelemetry GenAI tracing + the per-call cost the spec omits.

OpenTelemetry standardizes token usage but not dollars. This module emits
`gen_ai.*` spans and adds a `gen_ai.usage.cost` attribute from a small,
overridable pricing map.

Fully no-op unless OpenTelemetry is installed. We emit spans; we never build a
backend. Install with `pip install 'justllm[otel]'` and configure your own
exporter/collector.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

# (input, output) USD per 1M tokens. Substring-matched against the model name.
# Example rates — override with set_prices() for accuracy.
_PRICES: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-opus-4-8": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def set_prices(prices: Dict[str, Tuple[float, float]]) -> None:
    """Add or override (input, output) USD-per-1M-token rates by model substring."""
    _PRICES.update(prices)


def _price_for(model: str) -> Optional[Tuple[float, float]]:
    name = model.split("/")[-1].lower()
    for key, rate in _PRICES.items():
        if key in name:
            return rate
    return None


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    rate = _price_for(model)
    if rate is None:
        return None
    return prompt_tokens / 1e6 * rate[0] + completion_tokens / 1e6 * rate[1]


def _tracer():
    try:
        from opentelemetry import trace
    except Exception:
        return None
    return trace.get_tracer("justllm")


class _Recorder:
    """Records usage + cost onto a span once the response is available."""

    def __init__(self, span, model: str) -> None:
        self._span = span
        self._model = model

    def record(self, resp) -> None:
        if self._span is None:
            return
        try:
            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self._span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            self._span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
            cost = cost_of(self._model, prompt_tokens, completion_tokens)
            if cost is not None:
                self._span.set_attribute("gen_ai.usage.cost", round(cost, 6))
        except Exception:
            pass


@contextmanager
def call_span(model: str, op: str = "chat") -> Iterator[_Recorder]:
    tracer = _tracer()
    if tracer is None:
        yield _Recorder(None, model)
        return
    with tracer.start_as_current_span(f"gen_ai.{op}") as span:
        try:
            span.set_attribute("gen_ai.operation.name", op)
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute(
                "gen_ai.system", model.split("/")[0] if "/" in model else "unknown"
            )
        except Exception:
            pass
        yield _Recorder(span, model)
