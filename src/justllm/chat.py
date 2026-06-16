"""A small multi-turn conversation wrapper.

Holds the message history so you don't have to, and keeps the client's defaults
(fallback, caching, compression, routing) on every turn. Tool-using
conversations are the agent's job; this is for plain back-and-forth.

    chat = llm.chat(system="You are concise.")
    chat.send("Who wrote Dune?")
    chat.send("And the sequel?")   # remembers the first turn
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional


class Chat:
    def __init__(self, llm: Any, *, system: Optional[str] = None) -> None:
        self.llm = llm
        self.messages: List[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def send(self, content: str, **kwargs: Any) -> str:
        self.messages.append({"role": "user", "content": content})
        reply = self.llm._reply(self.messages, content, **kwargs)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    async def asend(self, content: str, **kwargs: Any) -> str:
        self.messages.append({"role": "user", "content": content})
        reply = await self.llm._areply(self.messages, content, **kwargs)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def stream(self, content: str, **kwargs: Any) -> Iterator[str]:
        """Stream the reply, then store it in history. No mid-stream failover."""
        from .compress import compress as _compress
        from .transports import stream as _stream

        self.messages.append({"role": "user", "content": content})
        messages = _compress(self.messages).messages if self.llm.compress else self.messages
        model = self.llm._primary_model(content)
        parts: List[str] = []
        for chunk in _stream(model, messages, cache=self.llm.cache, **kwargs):
            parts.append(chunk)
            yield chunk
        self.messages.append({"role": "assistant", "content": "".join(parts)})

    def reset(self) -> None:
        """Clear the history, keeping the system message if there was one."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = self.messages[:1]
        else:
            self.messages = []
