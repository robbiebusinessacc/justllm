# justllm benchmarks

The suite that proves the README's claims. Three things get measured:

| Benchmark | Proves | Needs network? |
|---|---|---|
| `bench_savings` | token + $ saved by compressing dynamic context | no |
| `bench_overhead` | the compression layer is thin (ms/call) | no |
| `bench_reliability` | fallback recovers provider failures | no |

## Run

```bash
pip install -e '.[benchmarks]'   # tiktoken + headroom for real numbers
python -m benchmarks.run
```

Everything runs **offline**. If `tiktoken` or `headroom` are not installed, the
suite still runs using fallbacks (a chars/4 token estimate and a whitespace-only
compressor) so it is never a hard blocker — but the savings numbers will be
understated, because the conservative fallback is *not* the real engine. Install
the extras for numbers worth quoting.

## Measured (headroom-ai 0.25.0, gpt-4o token basis)

| fixture (as a tool result) | tokens saved | % |
|---|---|---|
| `json_api_response` | 6,609 | 53.4% |
| `server_logs` | 6,706 | 97.1% |
| `rag_chunks` (small) | 0 | 0.0% |
| `code_file` (small) | 0 | 0.0% |
| **total** | **13,315** | **64.3%** |

## Caveats (read before quoting any number)

- **Compression only touches tool / retrieved content.** Headroom's router
  *protects* user and assistant turns (you never want to mangle the actual
  prompt). The fixtures are therefore wrapped as `role: "tool"` messages — which
  is how an agent receives this data anyway. Put bloat in a user message and you
  will (correctly) see 0% saved.
- **Headroom safely reverts when compression would inflate tokens** — on small
  or incompressible payloads you'll see `Optimization inflated tokens ...
  reverting to original` on stderr and 0% saved. Savings are never negative.
- **The real package is `headroom-ai`** (imports as `headroom`); plain
  `headroom` on PyPI is an unrelated CLI tool. Requires Python ≥3.10.
- **`headroom-ai[all]` pulls ML dependencies** and downloads a tokenizer/model
  from the HuggingFace Hub on first run (set `HF_TOKEN` to avoid rate limits).
- **Token counts** use tiktoken's `o200k_base` as a model-agnostic approximation.
- **Prices** in `bench_savings.PRICES` are *examples* — verify current rates.
- **Quality preservation is not yet measured here.** Headroom is reversible by
  design, but an end-to-end "does the answer stay correct after compression"
  benchmark (LLM-judge over real calls) is the next addition once the provider
  transport lands.
