# justllm

[![PyPI](https://img.shields.io/pypi/v/justllm)](https://pypi.org/project/justllm/)
[![CI](https://github.com/robbiebusinessacc/justllm/actions/workflows/ci.yml/badge.svg)](https://github.com/robbiebusinessacc/justllm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/justllm)](https://pypi.org/project/justllm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Production LLM calls. Just the three lines.**

![justllm demo](assets/demo.gif)

```python
from justllm import LLM

llm = LLM("anthropic/claude-opus-4-8")
llm("Summarize this contract.")
```

That call already does the work you'd normally wire up yourself, on by default:

- **Context compression.** [Headroom](https://github.com/chopratejas/headroom) shrinks tool output by 50–95% before it reaches the model.
- **Prompt-cache optimization.** Cache breakpoints go where each provider wants them (Anthropic, OpenAI, Google).
- **Reliability.** Calls retry with backoff, then fail over to the next provider.

```bash
pip install 'justllm[all]'
```

## A little more

Same three lines. Each of these is one call or one kwarg:

```python
llm.extract(Invoice, text)                    # structured output (validated Pydantic)
llm.stream("...")                             # token streaming
await llm.acall("...")                        # async
llm.map(prompts, concurrency=8)               # many prompts at once, in order
llm.embed(texts)                              # embeddings
llm.agent(system="...").run("...")            # tool-calling loop
LLM(router=Cascade(small=cheap, large=big))   # cheap first, escalate when needed
```

A few more things sit behind opt-in extras: OpenTelemetry traces that include the
per-call dollar cost (most setups leave that out), Langfuse-backed prompts,
semantic cascade escalation, and exact-match caching. The hard parts are already
wired; you just call them.

Runnable recipes: **[cookbook](examples/)**

## Why

The ecosystem splits two ways. You can have powerful but heavy (LiteLLM,
LangChain), or simple but thin (aisuite, any-llm). justllm sits in the middle:
every optimization is on, and the surface stays at three lines. Keeping it that
small was most of the work.

| | justllm | LiteLLM | aisuite |
|---|---|---|---|
| three-line call | yes | yes | yes |
| cross-provider fallback | on by default | config | no |
| context compression | on by default (Headroom) | manual trim | no |
| prompt-cache optimization | on by default | passthrough | no |
| structured output | yes (instructor) | passthrough | no |
| tool-calling agent | yes (minimal) | no | no |
| surface area | tiny | large | tiny |

It runs on LiteLLM underneath, so think of it as the opinionated layer on top
rather than a replacement.

---

*Alpha. The wiring is tested on CI (Python 3.10–3.13) and the call paths are
checked against live models.*

[Cookbook](examples/) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [MIT](LICENSE)
