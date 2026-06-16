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

## Shipped — 0.3.0

- Quality cascade routing (`Cascade`) — cheap model first, escalate only when the
  answer looks inadequate (default heuristic, or your own predicate). No judge call.
- Prompt-loader seam — `prompts.load(name, **vars)` from files, pluggable source
  via `set_loader()`, no registry built in.

## Next

- **CCR retrieve tool** — let an agent fetch the uncompressed original of a tool
  result on demand. Deferred from 0.3.0: Headroom exposes this through its
  proxy/MCP server, not the `compress()` library API, so it needs that mode (or a
  thin store) wired in before it fits the library cleanly.
- **Quality routing (pre-call)** — RouteLLM-style classify-then-route on the
  prompt, before any call (complements the post-answer `Cascade`). Opt-in.

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
