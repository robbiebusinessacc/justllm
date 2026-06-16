"""A minimal tool-calling loop.

An agent is a while loop: call the model, run any tool calls it asks for, feed
the results back, repeat until it stops asking for tools or the step cap is hit.
No graph DSL, no personas, no crews — just the loop people actually like. Tool
outputs are run through compression on the way back in, which is exactly where
justllm's compression earns its keep.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _tool_schema(fn: Callable) -> dict:
    """Derive an OpenAI-style tool schema from a function's signature + docstring."""
    sig = inspect.signature(fn)
    props: Dict[str, dict] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        props[name] = {"type": _PY_TO_JSON.get(param.annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": (fn.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


class Agent:
    def __init__(
        self,
        llm,
        *,
        system: Optional[str] = None,
        max_steps: int = 8,
        tools: Optional[List[Callable]] = None,
    ) -> None:
        self.llm = llm
        self.system = system
        self.max_steps = max_steps
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[dict] = []
        for fn in tools or []:
            self.tool(fn)

    def tool(self, fn: Callable) -> Callable:
        """Register a function as a tool (usable as a decorator)."""
        self._tools[fn.__name__] = fn
        self._schemas.append(_tool_schema(fn))
        return fn

    def _compress_result(self, content: str) -> str:
        if not self.llm.compress:
            return content
        try:
            from .compress import compress as _compress

            out = _compress([{"role": "tool", "content": content}])
            value = out.messages[0].get("content", content)
            return value if isinstance(value, str) else content
        except Exception:
            return content

    def run(self, prompt: str, **kwargs: Any) -> str:
        from .transports import _require_litellm

        litellm = _require_litellm()
        messages: List[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        model = self.llm._primary_model(prompt)

        for _ in range(self.max_steps):
            resp = litellm.completion(
                model=model,
                messages=messages,
                tools=self._schemas or None,
                **kwargs,
            )
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content or ""

            # Record the assistant turn that requested the tools.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn = self._tools.get(name)
                result = fn(**args) if fn else f"error: unknown tool {name!r}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": self._compress_result(str(result)),
                    }
                )

        return "error: max_steps reached without a final answer"
