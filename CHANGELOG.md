# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- See [ROADMAP.md](ROADMAP.md).

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

[Unreleased]: https://github.com/robbiebusinessacc/justllm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.2.0
[0.1.0]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.1.0
[0.0.1]: https://github.com/robbiebusinessacc/justllm/releases/tag/v0.0.1
