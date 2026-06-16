# justllm

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
```

## Status

Alpha (`0.1.0`). All wiring is unit-tested — transport, caching, fallback,
structured output, and the agent loop are verified with mocked providers, so
there's no network in CI:

- **Calls** — `llm("...")` and `llm.extract(Model, ...)` make real calls through
  LiteLLM, wrapped in cross-provider fallback.
- **Reliability** — `with_fallback` + `RetryPolicy`: retry-with-jitter on
  retryable errors only, one retry layer.
- **Caching** — native prompt caching (Anthropic breakpoint / OpenAI automatic)
  plus an opt-in exact-match cache.
- **Compression** — `compress` over Headroom; agent tool outputs are compressed
  automatically.
- **Agent** — a minimal tool-calling loop with a hard step cap.

Not yet validated against live provider APIs end-to-end (that needs keys — see
`benchmarks/bench_e2e.py`). Treat live behavior as alpha.

## Benchmarks

```bash
pip install -e '.[benchmarks]'
python -m benchmarks.run
```

Measures token/cost savings from compression, the overhead the layer adds, and
that fallback actually recovers provider failures. The suite runs even without
the optional deps (using fallbacks), so it is never a hard blocker.

## License

MIT © Robert Walmsley
