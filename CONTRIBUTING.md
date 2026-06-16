# Contributing to justllm

Thanks for being here. justllm aims to be the **simple middle** of the LLM
tooling world — production defaults (fallback, caching, compression) behind a
three-line API. Contributions are very welcome, and the bar is simple: keep it
**state-of-the-art *and* easy to use** at the same time.

That dual goal is the whole project. The design principles below are how we
protect it — please read them before opening a PR.

## Design principles

These are non-negotiable; a PR that violates one will be asked to change.

1. **Tiny surface.** Every public symbol must earn its place. The common case is
   three lines. If a feature adds a new top-level concept, it had better delete
   one too — or live behind a method/extra most users never see.
2. **Opinionated, safe defaults.** The right thing happens with zero config.
   Knobs are escape hatches, not the main path. A default must never be lossy or
   correctness-risky (this is why semantic caching is opt-in, never the silent
   meaning of `cache`).
3. **Don't reinvent the wheel.** We are the *opinionated layer*, not a provider
   SDK. Wrap best-in-class tools — LiteLLM (transport), instructor (structured
   output), Headroom (compression) — behind one clean surface. If LiteLLM
   already does it well, we document it, we don't reimplement it.
4. **Every feature justifies its complexity.** Judge by impact-per-complexity.
   "It would be cool" is not enough. "It saves real money / prevents real
   failures with little surface cost" is.
5. **Tested.** Wiring is unit-tested with mocked providers (no network in CI).
   Live behavior is validated with `benchmarks/bench_e2e.py` against a real
   model (a free Groq key or local Ollama both work).
6. **Honest numbers.** Benchmarks state their basis and caveats. No cherry-picked
   or fabricated savings.

If you're unsure whether an idea fits, open a
[Discussion](https://github.com/robbiebusinessacc/justllm/discussions) first —
that's cheaper than a rejected PR.

## Dev setup

```bash
git clone https://github.com/robbiebusinessacc/justllm
cd justllm
python -m venv .venv && source .venv/bin/activate    # Python 3.10+
pip install -e '.[all,dev]'
```

## Before you push

```bash
pytest -q          # all wiring tests must pass
ruff check .       # lint must be clean
```

Optional, to see the value props measured:

```bash
python -m benchmarks.run                       # compression + reliability (offline)
python -m benchmarks.bench_e2e                 # real call (uses a key or local Ollama)
```

## Good places to start

- **Add/verify a provider.** Transport rides LiteLLM, so most providers already
  work — confirm one end-to-end and add it to the docs/model list.
- **Add a benchmark fixture.** Real-world tool-output shapes in
  `benchmarks/fixtures.py` make the compression numbers more representative.
- **Docs & cookbook examples.** Short, runnable recipes.
- **Pick up a `good first issue`.** They're scoped to be self-contained.

See [ROADMAP.md](ROADMAP.md) for where the project is headed — and the explicit
**non-goals**, so you don't spend effort on something we won't merge.

## PR process

1. Branch off `main`, keep the change small and focused.
2. Add tests; run `pytest -q` and `ruff check .`.
3. Fill in the PR checklist (it mirrors the design principles).
4. A maintainer reviews for correctness *and* for surface/simplicity.

By contributing you agree your work is licensed under the project's
[MIT License](LICENSE).
