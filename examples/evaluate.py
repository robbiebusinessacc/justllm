"""Evaluate: grade an output with an LLM judge, or score a whole test set.

The judge needs a reasonably capable model and `pip install 'justllm[structured]'`.
The custom-scorer path is plain Python and runs anywhere.

    GROQ_API_KEY=gsk_... JUSTLLM_MODEL=groq/llama-3.1-8b-instant python examples/evaluate.py
"""
import os

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


def main() -> None:
    llm = LLM(MODEL)

    # 1. LLM-as-judge: score one output against a rubric.
    v = llm.judge(
        "The capital of France is Paris.",
        criteria="Is this factually correct and on topic?",
    )
    print(f"judge -> passed={v.passed} score={v.score}  ({v.reasoning[:60]}...)")

    # 2. Score a test set. Cases with an 'input' are run through the model first;
    #    the custom scorer here skips the judge and just checks the text.
    cases = [
        {"input": "In one word, the capital of Japan?", "expected": "tokyo"},
        {"input": "In one word, the capital of Italy?", "expected": "rome"},
    ]
    report = llm.evaluate(
        cases, scorer=lambda out, case: case["expected"] in out.lower()
    )
    print("evaluate ->", report.summary())


if __name__ == "__main__":
    main()
