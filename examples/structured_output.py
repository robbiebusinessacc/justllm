"""Structured output: get a validated Pydantic object back.

Needs a capable model and `pip install 'justllm[structured]'`.

    GROQ_API_KEY=gsk_... JUSTLLM_MODEL=groq/llama-3.1-8b-instant \
        python examples/structured_output.py
"""
import os

from pydantic import BaseModel

from justllm import LLM

MODEL = os.getenv("JUSTLLM_MODEL", "openai/gpt-4o")


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


def main() -> None:
    llm = LLM(MODEL)
    invoice = llm.extract(Invoice, "Invoice: Acme Corp billed us $4,200 USD.")
    print(invoice)
    print("total is a float:", isinstance(invoice.total, float))


if __name__ == "__main__":
    main()
