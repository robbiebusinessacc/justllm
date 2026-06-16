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

You don't call any of these yourself; they run inside `llm(...)`. (The demo calls
`compress()` directly just to show the savings number. Normally you never touch it.)

```bash
pip install 'justllm[all]'
```

## More, when you need it

You set up `llm` once (those three lines). After that, each of these is a single
call on it. Reach for the ones you need and ignore the rest:

```python
llm.stream("...")                    # token streaming
await llm.acall("...")               # async
llm.map(prompts, concurrency=8)      # many prompts at once, in order
llm.extract(Invoice, text)           # structured output (validated Pydantic)
llm.chat()                           # multi-turn, keeps history
llm.agent(system="...").run("...")   # tool-calling loop
llm.judge(output, criteria="...")    # LLM-as-judge score
llm.evaluate(cases)                  # run + grade a test set
```

Also there, all opt-in: `llm.embed(...)`, routing (`Router` and `Cascade`),
OpenTelemetry traces with the per-call dollar cost, Langfuse-backed prompts, and
exact-match caching. Runnable versions of everything are in the
[cookbook](examples/).

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
