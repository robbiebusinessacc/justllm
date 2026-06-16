"""Compression: shrink bulky tool output before it reaches the model.

No API call; this just runs Headroom over a tool message and reports what it saved.
For real numbers: `pip install 'justllm[compression,benchmarks]'` (tiktoken gives
accurate token counts; without it, an estimate is used).
"""
import json

from justllm import compress


def main() -> None:
    # A verbose tool result, the kind of thing that bloats agent context.
    tool_output = json.dumps(
        {"users": [{"id": i, "name": f"user{i}", "active": True} for i in range(80)]},
        indent=2,
    )
    # Headroom compresses tool/retrieved content; it protects user/assistant turns.
    messages = [{"role": "tool", "tool_call_id": "c1", "content": tool_output}]

    result = compress(messages, model="gpt-4o")
    print(
        f"tokens: {result.tokens_before} -> {result.tokens_after} "
        f"({result.pct_saved:.1f}% saved)"
    )


if __name__ == "__main__":
    main()
