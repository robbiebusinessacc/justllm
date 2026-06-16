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

Cross-provider fallback, prompt-cache optimization, and context compression are
**on by default**. No config. Outgrow it? Drop down to LiteLLM.

```bash
pip install 'justllm[all]'
```

## A little more

```python
llm.extract(Invoice, text)                   # validated Pydantic, not a string
llm.stream("...")                            # token streaming
await llm.acall("...")                       # async
llm.agent(system="...").run("...")           # tool-calling loop
LLM(router=Cascade(small=cheap, large=big))  # cheap first, escalate when needed
```

Runnable recipes for all of it: **[cookbook →](examples/)**

## Why

The ecosystem is split: powerful but heavy (LiteLLM, LangChain), or simple but
thin (aisuite, any-llm). justllm is the middle — production defaults behind a
three-line surface. The discipline *is* the product.

---

*Alpha. Wiring is tested on CI (Python 3.10–3.13); call paths are validated live.*

[Cookbook](examples/) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [MIT](LICENSE)
