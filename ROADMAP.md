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

## Shipped — 0.2.0

- Streaming (`llm.stream`), async (`llm.acall` / `llm.aextract` / `awith_fallback`)
- Opt-in length-based routing (`Router`)
- OpenTelemetry GenAI spans **with the per-call `gen_ai.usage.cost`** the spec omits
- Headroom's per-provider cache optimizer (Anthropic/OpenAI/Google), replacing the
  hand-rolled breakpoint
- `CompressConfig` passthrough to tune compression

## Next — 0.3.0

- **Quality cascade routing** — send to a cheap model first, escalate to a strong
  one only when needed (RouteLLM-style). Builds on the length-based `Router`.
  Opt-in, never a surprise default.
- **CCR retrieve tool** — wire Headroom's Compress-Cache-Retrieve so an agent can
  fetch the full, uncompressed original of a tool result on demand.
- **Prompt loader seam** — load prompts from files / a registry (Langfuse-style)
  without building a registry ourselves.

## Later

- **Semantic routing** for intent dispatch (opt-in).
- **Streaming inside the agent loop** and richer streamed tool-call handling.
- **More provider recipes** in the cookbook.

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
