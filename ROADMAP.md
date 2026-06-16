# Roadmap

justllm's North Star: **the simple middle** — production-grade defaults behind a
three-line API. Everything here is filtered through the
[design principles](CONTRIBUTING.md#design-principles). Dates are intentionally
absent; this is priority order, not a schedule. Ideas and PRs welcome — open a
[Discussion](https://github.com/robbiebusinessacc/justllm/discussions).

## Shipped — 0.1.0

- One client, OpenAI-shaped, across providers (via LiteLLM)
- Cross-provider fallback + retry-with-jitter (one retry layer)
- Native prompt caching (Anthropic breakpoint / OpenAI automatic) + opt-in exact cache
- Reversible context compression on tool output (via Headroom)
- Structured output — `extract()` returns a validated Pydantic instance (via instructor)
- Minimal tool-calling agent loop with a hard step cap
- Benchmark suite (compression, reliability, end-to-end) + mock-tested wiring

## Next — 0.2.0 (highest impact-per-complexity)

- **Streaming** — `llm.stream("...")` yielding tokens. Table stakes for chat UIs.
- **Async** — `await llm.acall(...)` / `aextract(...)`. LiteLLM already supports it.
- **Observability + cost layer** — emit OpenTelemetry GenAI spans, and add the
  per-call **dollar cost** attribute the OTel spec omits. Off unless a collector
  is configured. (The research-identified gap nobody fills.)

## Later — 0.3.0

- **Smart routing (opt-in)** — cheap-model-first cascade / quality routing
  (RouteLLM-style), and semantic routing for intent dispatch. Opt-in, never a
  surprise default.
- **Prompt loader seam** — load prompts from files / a registry (Langfuse-style)
  without building a registry ourselves.
- **Compression tuning** — expose Headroom config (model limits, per-type
  aggressiveness) through one clean knob.

## Explicit non-goals (the traps)

We will **not** merge these — they break "simple *and* safe," or reinvent a
hosted product. Documented so contributors don't waste effort:

- A graph/DAG DSL or heavyweight multi-agent "crews" as the core abstraction
- Semantic caching **on by default** (silently returns wrong answers)
- Neural prompt compression (e.g. LLMLingua) in the default call path
- Building our own observability **backend**, prompt **registry**, or UI
- Re-implementing provider SDKs we can wrap

If you want one of these, it almost certainly belongs in a separate package that
builds *on* justllm.

## How to influence this

Open a Discussion or a feature request (the template asks "how does this stay
simple?" on purpose). The roadmap follows real, measured needs.
