"""A small LLM-as-judge evaluation layer.

Deliberately thin. A judge is just a structured-output call with a rubric, so it
rides `extract()`; the test-set runner rides `map()`. No new engine, no heavy
framework. For specialized metrics (RAG faithfulness, etc.) reach for deepeval or
RAGAS — this covers the common case: "score this output" and "score this set".

Needs `pip install 'justllm[structured]'` (the judge uses structured output).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from pydantic import BaseModel


class Verdict(BaseModel):
    # Field order matters: the model reasons first, then scores, then decides.
    reasoning: str
    score: int
    passed: bool


@dataclass
class CaseResult:
    output: str
    passed: bool
    score: Optional[float]
    input: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class EvalReport:
    results: List[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def mean_score(self) -> Optional[float]:
        scores = [r.score for r in self.results if r.score is not None]
        return sum(scores) / len(scores) if scores else None

    def summary(self) -> str:
        n = len(self.results)
        line = f"{sum(r.passed for r in self.results)}/{n} passed ({self.pass_rate:.0%})"
        if self.mean_score is not None:
            line += f", mean score {self.mean_score:.2f}"
        return line


def judge(
    llm: Any,
    output: str,
    *,
    criteria: str,
    reference: Optional[str] = None,
    scale: int = 5,
    **kwargs: Any,
) -> Verdict:
    """Grade one output against a criterion. Returns a `Verdict`."""
    parts = [
        "You are an impartial grader. Grade the output against the criterion.",
        f"Criterion: {criteria}",
    ]
    if reference is not None:
        parts.append(f"Reference answer:\n{reference}")
    parts.append(f"Output to grade:\n{output}")
    parts.append(
        f"Reason briefly, give a score from 1 to {scale} (higher is better), and "
        "a pass/fail decision."
    )
    return llm.extract(Verdict, "\n\n".join(parts), **kwargs)


def evaluate(
    llm: Any,
    cases: Any,
    *,
    scorer: Optional[Callable[[str, dict], Any]] = None,
    grader: Any = None,
    concurrency: int = 8,
    **kwargs: Any,
) -> EvalReport:
    """Run and score a list of cases.

    Each case is a dict with an ``output`` (to grade as-is) or an ``input`` (run
    through the model first), plus a ``criteria`` (for the default judge) and an
    optional ``reference``. Pass a ``scorer(output, case) -> bool | float`` to
    skip the LLM judge and score programmatically; pass ``grader=`` to judge with
    a different model than the one under test.
    """
    cases = list(cases)

    # Generate outputs for cases that only specify an input (concurrently).
    need_gen = [(i, c["input"]) for i, c in enumerate(cases) if "output" not in c and "input" in c]
    generated = {}
    if need_gen:
        outs = llm.map([inp for _, inp in need_gen], concurrency=concurrency, **kwargs)
        for (i, _), out in zip(need_gen, outs, strict=True):
            generated[i] = out

    grader = grader or llm
    results: List[CaseResult] = []
    for i, case in enumerate(cases):
        if "output" in case:
            output = case["output"]
        elif i in generated:
            output = generated[i]
        else:
            raise ValueError("each case needs an 'output' or an 'input'")

        if scorer is not None:
            raw = scorer(output, case)
            if isinstance(raw, bool):
                passed, score = raw, 1.0 if raw else 0.0
            else:
                score = float(raw)
                passed = score >= 0.5
            reasoning = None
        else:
            verdict = grader.judge(
                output,
                criteria=case.get("criteria", "Is the answer correct and helpful?"),
                reference=case.get("reference"),
            )
            passed, score, reasoning = verdict.passed, float(verdict.score), verdict.reasoning

        results.append(
            CaseResult(
                output=output,
                passed=passed,
                score=score,
                input=case.get("input"),
                reasoning=reasoning,
            )
        )

    return EvalReport(results)
