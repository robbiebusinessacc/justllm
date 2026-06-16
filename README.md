# justllm

[![PyPI](https://img.shields.io/pypi/v/justllm)](https://pypi.org/project/justllm/)
[![CI](https://github.com/robbiebusinessacc/justllm/actions/workflows/ci.yml/badge.svg)](https://github.com/robbiebusinessacc/justllm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/justllm)](https://pypi.org/project/justllm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Production LLM calls. Just the three lines.**

```python
from justllm import LLM

llm = LLM("anthropic/claude-opus-4-8")
llm("Summarize this contract.")
```

That one call already does what you'd normally wire by hand — **on by default,
zero config:**

- **Context compression** — [Headroom](https://github.com/chopratejas/headroom) shrinks tool output 50–95% before it reaches the model
- **Prompt-cache optimization** — provider-optimal cache breakpoints (Anthropic / OpenAI / Google)
- **Reliability** — retry with backoff, then fail over across providers

```bash
pip install 'justllm[all]'
```

## More, when you want it

Same three-line surface — each of these is one call or one kwarg:

```python
llm.extract(Invoice, text)                    # structured output (validated Pydantic)
llm.stream("...")                             # token streaming
await llm.acall("...")                        # async
llm.agent(system="...").run("...")            # tool-calling loop
LLM(router=Cascade(small=cheap, large=big))   # cheap first, escalate when needed
```

Plus OpenTelemetry tracing with the per-call **cost** the spec omits (`[otel]`),
Langfuse-backed prompts, semantic cascade escalation, exact-match caching — all
opt-in. The point: every one of these is SOTA under the hood and a one-liner on top.

Runnable recipes for all of it: **[cookbook →](examples/)**

## Why

The ecosystem is split: powerful but heavy (LiteLLM, LangChain), or simple but
thin (aisuite, any-llm). justllm is the middle — every optimization on, behind a
three-line surface. The discipline *is* the product.

| | justllm | LiteLLM | aisuite |
|---|---|---|---|
| three-line call | yes | yes | yes |
| cross-provider fallback | on by default | config | no |
| context compression | on by default (Headroom) | manual trim | no |
| prompt-cache optimization | on by default | passthrough | no |
| structured output | yes (instructor) | passthrough | no |
| tool-calling agent | yes (minimal) | no | no |
| surface area | tiny | large | tiny |

justllm builds *on* LiteLLM for transport — it's the opinionated layer on top,
not a replacement for it.

---

*Alpha. Wiring is tested on CI (Python 3.10–3.13); call paths are validated live.*

[Cookbook](examples/) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [MIT](LICENSE)
