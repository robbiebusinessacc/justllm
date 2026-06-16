# justllm cookbook

Short, runnable recipes. Each file stands alone and is guarded by
`if __name__ == "__main__"`, so you can run any of them directly.

## Running

Most recipes read the model from `JUSTLLM_MODEL` (default `openai/gpt-4o`), so
you can point them at whatever you have:

```bash
pip install 'justllm[all]'

# a hosted model
OPENAI_API_KEY=sk-...  python examples/basics.py
GROQ_API_KEY=gsk_...   JUSTLLM_MODEL=groq/llama-3.1-8b-instant python examples/agent_with_tools.py

# or a free local model, no key needed
JUSTLLM_MODEL=ollama_chat/llama3.2:1b python examples/streaming.py
```

## Recipes

| File | Shows |
|---|---|
| [basics.py](basics.py) | a call, a cross-provider fallback chain, caching |
| [streaming.py](streaming.py) | token streaming with `llm.stream` |
| [async_calls.py](async_calls.py) | `await llm.acall(...)` |
| [structured_output.py](structured_output.py) | `llm.extract(Model, ...)` → a validated Pydantic object |
| [agent_with_tools.py](agent_with_tools.py) | a tool-calling agent loop |
| [compression.py](compression.py) | shrink tool output before it hits the model (offline) |
| [routing.py](routing.py) | length-based `Router` and cheap-first `Cascade` |
| [prompts.py](prompts.py) | file loader, hot-reload, and a Langfuse adapter (offline) |
| [observability.py](observability.py) | OpenTelemetry spans with per-call cost |

Capability notes: `structured_output` and `agent_with_tools` need a reasonably
capable model (tiny local models can't do structured output or tool calls
reliably). `compression` and `prompts` make no API calls.
