# justllm cookbook

Short, self-contained recipes. Each file runs on its own, so grab whichever one
matches what you're trying to do.

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
| [demo.py](demo.py) | a 30-second tour (call + compression + streaming), recording-ready |
| [basics.py](basics.py) | a call, a cross-provider fallback chain, caching |
| [streaming.py](streaming.py) | token streaming with `llm.stream` |
| [async_calls.py](async_calls.py) | `await llm.acall(...)` |
| [batch.py](batch.py) | run many prompts at once with `llm.map(...)` |
| [chat.py](chat.py) | a multi-turn conversation that remembers history |
| [structured_output.py](structured_output.py) | `llm.extract(Model, ...)` → a validated Pydantic object |
| [agent_with_tools.py](agent_with_tools.py) | a tool-calling agent loop |
| [compression.py](compression.py) | shrink tool output before it hits the model (offline) |
| [routing.py](routing.py) | length-based `Router` and cheap-first `Cascade` |
| [prompts.py](prompts.py) | file loader, hot-reload, and a Langfuse adapter (offline) |
| [observability.py](observability.py) | OpenTelemetry spans with per-call cost |

Two caveats. `structured_output` and `agent_with_tools` need a reasonably capable
model; the tiny local ones won't handle structured output or tool calls well.
`compression` and `prompts` don't call any API, so they run anywhere.
