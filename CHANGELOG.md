# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- See [ROADMAP.md](ROADMAP.md).

## [0.7.0] — 2026-06-16

### Added
- Evaluation layer (LLM-as-judge), built on structured output — thin by design:
  - `llm.judge(output, criteria=...)` returns a `Verdict` (reasoning, score,
    pass/fail).
  - `llm.evaluate(cases)` runs and grades a test set concurrently (reuses `map`):
    the LLM judge by default, or a `scorer(output, case)` function for
    programmatic checks; pass `grader=` to judge with a different model.
  - `Verdict` and `EvalReport` live in `justllm.eval`. Needs `[structured]`.

### Changed
- CI type-check installs pydantic so mypy sees the eval models' types.

## [0.6.0] — 2026-06-16

### Added
- `llm.chat(system=...)` returns a `Chat` that remembers history across turns:
  `chat.send(...)`, async `chat.asend(...)`, streaming `chat.stream(...)`, and
  `chat.reset()`. Same defaults as the client (fallback, caching, compression,
  routing). No more hand-managing a `messages` list for back-and-forth.

### Changed
- CI now type-checks with mypy (so `py.typed` is backed by a real check). Fixed
  two internal type issues; no behavior or public-API change.
- Refactored single-call and chat paths onto one shared engine; behavior of
  existing calls is unchanged.

## [0.5.0] — 2026-06-16

### Added
- `llm.map(prompts, concurrency=8)` (and async `amap`): run many prompts at once
  with a bounded concurrency limit; results come back in order.
- `llm.embed(texts)`: embeddings via LiteLLM (one vector for a string, a list for
  a list). Defaults to a sensible embedding model for OpenAI/Google; pass `model=`
  for others.
- `py.typed` marker so type checkers (mypy, pyright) pick up the library's hints.

## [0.4.0] — 2026-06-16

### Added
- `embedding_escalator(...)`: an embedding-based escalation predicate for
  `Cascade`. Escalates when the cheap model's answer is semantically close to a
  refusal/hedge — catching paraphrases ("it's hard to say", "not enough
  information") that the keyword heuristic misses. Defaults to local fastembed
  (bge-small ONNX, no torch); `[embeddings]` extra. Detects hedging, not factual
  errors.
- Cookbook: runnable recipes in `examples/` with a CI compile check (docs only;
  not part of the published package).

## [0.3.2] — 2026-06-15

### Added
- `prompts.file_loader(base_dir, cache=True, on_reload=...)`: a hot-reloading
  file loader for the prompt seam — caches prompts and re-reads only when a file's
  mtime changes (no background watcher, no new dependencies). `cache=False` to
  always re-read; `on_reload(name, path)` to observe (re)loads.

## [0.3.1] — 2026-06-15

### Added
- `prompts.langfuse_loader(...)`: a ready Langfuse adapter for the prompt-loader
  seam. Fetches text prompts from Langfuse and renders them through the existing
  `set_loader` seam (uses Langfuse's `{{var}}` -> `{var}` conversion). Optional
  `[langfuse]` extra; text prompts only.

## [0.3.0] — 2026-06-15

### Added
- Quality cascade routing: `Cascade(small=..., large=..., escalate_if=...)` calls
  the cheap model first and escalates to the strong one only when the answer looks
  inadequate (default heuristic, or your own predicate). No judge LLM call.
- Prompt-loader seam: `prompts.load(name, **vars)` loads templates from files and
  renders only the variables you pass (literal braces preserved). Swap the source
  via `prompts.set_loader(...)`; no registry built in.

### Notes
- CCR (Headroom's retrieve-original tool) deferred: it requires Headroom's
  proxy/store mode, which doesn't fit the thin library layer. Still on the roadmap.
- Cascade and the prompt loader validated live against Ollama and Groq.

## [0.2.0] — 2026-06-15

### Added
- Streaming: `llm.stream(prompt)` yields text chunks.
- Async: `llm.acall(...)`, `llm.aextract(...)`, and `awith_fallback`.
- Opt-in routing: `Router` (length-based, deterministic) via `LLM(router=...)`.
- Observability: optional OpenTelemetry GenAI spans with a per-call
  `gen_ai.usage.cost` attribute and an overridable pricing map (`[otel]` extra).
- `compress()` accepts a Headroom `CompressConfig` to tune compression.

### Changed
- Cache optimization now uses Headroom's per-provider cache optimizer
  (multi-breakpoint, 1024-token minimum, prefix stabilization) covering
  Anthropic, OpenAI, and Google — replacing a hand-rolled single breakpoint.

### Notes
- New call paths validated live against Ollama and Groq.

## [0.1.0] — 2026-06-15

### Added
- Provider transport via LiteLLM — `llm("...")` makes real calls (optional
  `[litellm]` extra).
- Structured output — `llm.extract(Model, ...)` returns a validated Pydantic
  instance, via instructor (`[structured]` extra).
- Minimal tool-calling agent — `llm.agent()` with a hard step cap; tool outputs
  are compressed automatically.
- Native prompt caching (Anthropic breakpoint / OpenAI automatic) and an opt-in
  exact-match cache.
- End-to-end benchmark (`benchmarks/bench_e2e.py`) that auto-detects an API key
  or a local Ollama model.
- Optional dependency extras: `litellm`, `structured`, `compression`,
  `benchmarks`, `all`.

### Notes
- Wiring is unit-tested with mocked providers; live behavior validated against
  Ollama (`llama3.2:1b`) and Groq (`llama-3.1-8b-instant`).

## [0.0.1] — 2026-06-15

### Added
- Initial release: reliability layer (`with_fallback`, `RetryPolicy`) and the
  Headroom-backed compression adapter (`compress`), plus the benchmark scaffold.

[Unreleased]: https://github.com/robbiebusinessacc/justllm/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.7.0
[0.6.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.6.0
[0.5.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.5.0
[0.4.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.4.0
[0.3.2]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.3.2
[0.3.1]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.3.1
[0.3.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.3.0
[0.2.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.2.0
[0.1.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.1.0
[0.0.1]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.0.1
