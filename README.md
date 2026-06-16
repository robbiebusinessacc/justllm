# justllm

[![PyPI](https://img.shields.io/pypi/v/justllm)](https://pypi.org/project/justllm/)
[![CI](https://github.com/robbiebusinessacc/justllm/actions/workflows/ci.yml/badge.svg)](https://github.com/robbiebusinessacc/justllm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/justllm)](https://pypi.org/project/justllm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Production LLM calls. Just the three lines.**

```python
from justllm import LLM

llm = LLM("anthropic/claude-opus-4-8")
reply = llm("Summarize this contract.")
```

Cross-provider fallback, native prompt caching, and reversible context
compression are **on by default**. No config. The surface stays tiny on
purpose — the moment you need a dozen knobs, that is what LiteLLM is for.

## Why

The ecosystem split in two: feature-complete but heavy (LiteLLM, LangChain),
or simple but feature-thin (aisuite, any-llm). Nobody ships the production
layer behind a three-line front door. `justllm` is that middle.

The one number that makes it worth a switch: compressing the dynamic junk that
bloats agent calls — tool outputs, logs, RAG dumps — cuts the input-token bill
without touching your code. Measured here (gpt-4o token basis): **53% saved on a
JSON API tool result, 97% on repetitive logs**, with a safe no-op when
compression wouldn't help. The engine is
[Headroom](https://github.com/chopratejas/headroom) (PyPI: `headroom-ai`,
content-aware and reversible); justllm applies it only to tool/retrieved
content, never to your prompts. See [`benchmarks/`](benchmarks/).

## Install

```bash
pip install 'justllm[all]'     # transport + structured output + compression
```

Or take only what you need: `justllm[litellm]` (real calls), `justllm[structured]`
(`extract()`), `justllm[compression]` (Headroom). The bare `pip install justllm`
gives you the API and the reliability layer; calls raise a clear error until a
transport is installed.

## Usage

```python
# fallback chain + explicit knobs
llm = LLM(
    chain=["anthropic/claude-opus-4-8", "openai/gpt-5", "groq/llama-3.1-70b"],
    compress=True,     # reversible, dynamic-context only
    cache="prompt",    # "cache" never silently means semantic
)

# structured output — a validated Pydantic instance
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float

inv = llm.extract(Invoice, "Parse: Acme Corp billed $4,200")

# a minimal tool-calling agent (tool outputs are auto-compressed)
agent = llm.agent(system="You are a travel assistant.", max_steps=8)

@agent.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return weather_api(city)

agent.run("What should I pack for Boston this weekend?")

# streaming
for chunk in llm.stream("Tell me a short story."):
    print(chunk, end="")

# async (acall / aextract)
reply = await llm.acall("Summarize this.")

# opt-in routing: short prompts -> cheap model, long -> strong (no extra call)
from justllm import Router
routed = LLM(router=Router(small="groq/llama-3.1-8b-instant", large="openai/gpt-4o"))

# cheap-first cascade: escalate to the strong model only when needed
from justllm import Cascade
smart = LLM(router=Cascade(small="groq/llama-3.1-8b-instant", large="openai/gpt-4o"))

# load prompts from files (no registry); only your {vars} are substituted
from justllm import prompts
prompt = prompts.load("summary", document=text)   # reads prompts/summary.txt

# ...or back the same seam with a registry (Langfuse, etc.)
prompts.set_loader(prompts.langfuse_loader(label="production"))
prompt = prompts.load("summary", document=text)   # now fetched from Langfuse
```

Optional OpenTelemetry tracing (`pip install 'justllm[otel]'`) emits `gen_ai.*`
spans **with a per-call `gen_ai.usage.cost`** — the dollar figure the OTel spec
leaves out. No-op until you configure a collector.

## Status

Alpha (`0.3.0`). Wiring is unit-tested with mocked providers (no network in CI),
and the call paths are validated live against Ollama and Groq:

- **Calls** — sync `llm("...")`, async `llm.acall(...)`, and `llm.stream(...)`,
  all through LiteLLM and wrapped in cross-provider fallback.
- **Structured output** — `llm.extract(Model, ...)` / `await llm.aextract(...)`
  return a validated Pydantic instance (via instructor).
- **Reliability** — `with_fallback` / `awith_fallback` + `RetryPolicy`:
  retry-with-jitter on retryable errors only, one retry layer.
- **Caching** — Headroom's per-provider cache optimizer (Anthropic breakpoints;
  OpenAI/Google handled) plus an opt-in exact-match cache.
- **Compression** — `compress` over Headroom (tunable via `CompressConfig`);
  agent tool outputs are compressed automatically.
- **Routing** — opt-in `Router` (length-based) and `Cascade` (cheap-first,
  escalate only when needed); deterministic, no extra judge call.
- **Prompts** — `prompts.load(name, **vars)` file loader with a pluggable seam
  (swap in Langfuse etc.); no registry built in.
- **Observability** — optional OpenTelemetry GenAI spans with the per-call
  `gen_ai.usage.cost` the spec omits; no-op unless `[otel]` is installed.
- **Agent** — a minimal tool-calling loop with a hard step cap.

Live behavior is still alpha — exercise it with your own keys (or local Ollama)
via `benchmarks/bench_e2e.py`.

## Benchmarks

```bash
pip install -e '.[benchmarks]'
python -m benchmarks.run
```

Measures token/cost savings from compression, the overhead the layer adds, and
that fallback actually recovers provider failures. The suite runs even without
the optional deps (using fallbacks), so it is never a hard blocker.

## Contributing

Contributions are very welcome — the goal is to stay **SOTA and easy to use** at
the same time. Start with [CONTRIBUTING.md](CONTRIBUTING.md) (especially the
design principles that keep the surface small), see where things are headed in
[ROADMAP.md](ROADMAP.md), and look for `good first issue` labels.

## License

[MIT](LICENSE)
