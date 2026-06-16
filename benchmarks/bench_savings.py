"""Token + cost savings from context compression — the headline benchmark.

Runs offline: needs only `tiktoken` and (ideally) `headroom`. Without them it
still runs using fallbacks, with a clear caveat in the output.

    pip install -e '.[benchmarks]'
    python -m benchmarks.bench_savings
"""
from __future__ import annotations

from typing import Dict, List

from justllm.compress import compress

from .fixtures import all_fixtures

# Example input prices ($ per 1M input tokens). VERIFY against current rates
# before quoting these anywhere — provider pricing moves.
PRICES = {
    "gpt-4o": 2.50,
    "claude-opus-4-8": 15.00,
    "claude-haiku-4-5": 1.00,
}


def run(model: str = "gpt-4o") -> List[Dict]:
    rows = []
    for name, payload in all_fixtures().items():
        # Headroom protects user/assistant turns and compresses tool output, so
        # the bloat must arrive as a tool result — which is exactly how an agent
        # receives this data in practice.
        messages = [
            {"role": "user", "content": f"Use the {name} result to answer."},
            {"role": "tool", "tool_call_id": "call_1", "content": payload},
        ]
        result = compress(messages, model=model)
        rows.append(
            {
                "fixture": name,
                "before": result.tokens_before,
                "after": result.tokens_after,
                "pct_saved": result.pct_saved,
            }
        )
    return rows


def format_table(rows: List[Dict], model: str = "gpt-4o") -> str:
    price = PRICES.get(model.split("/")[-1], 0.0)
    line = "-" * 64
    out = [
        f"\nContext compression  (model={model}, tiktoken token basis)",
        line,
        f"{'fixture':<22}{'before':>10}{'after':>10}{'saved':>10}{'%':>10}",
    ]
    tb = ta = 0
    for r in rows:
        tb += r["before"]
        ta += r["after"]
        out.append(
            f"{r['fixture']:<22}{r['before']:>10}{r['after']:>10}"
            f"{r['before'] - r['after']:>10}{r['pct_saved']:>9.1f}%"
        )
    out.append(line)
    pct = 100.0 * (tb - ta) / tb if tb else 0.0
    out.append(f"{'TOTAL':<22}{tb:>10}{ta:>10}{tb - ta:>10}{pct:>9.1f}%")
    if price:
        per_1k = (tb - ta) / 1e6 * price * 1000
        out.append(
            f"\nAt ${price:.2f}/1M input tokens: "
            f"~${per_1k:,.2f} saved per 1,000 calls of this shape."
        )
    return "\n".join(out)


if __name__ == "__main__":
    print(format_table(run()))
