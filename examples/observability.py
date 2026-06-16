"""Observability: OpenTelemetry GenAI spans, including the per-call dollar cost.

Needs `pip install 'justllm[otel]'`. Here we export spans to the console; in
production you'd point an OTLP exporter at your collector instead.

    GROQ_API_KEY=gsk_... JUSTLLM_MODEL=groq/llama-3.1-8b-instant \
        python examples/observability.py
"""
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from justllm import LLM, observability

MODEL = os.getenv("JUSTLLM_MODEL", "groq/llama-3.1-8b-instant")


def main() -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    # Pricing the OTel spec leaves out. Override these with your real rates.
    observability.set_prices({"llama-3.1-8b-instant": (0.05, 0.08)})

    LLM(MODEL)("In one word, the capital of Italy?")
    # A `gen_ai.chat` span prints to the console, including gen_ai.usage.cost.


if __name__ == "__main__":
    main()
