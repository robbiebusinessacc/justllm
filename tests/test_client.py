"""Wiring tests for the call path: transport, caching, fallback, extract, agent.

LiteLLM/instructor are monkeypatched, so these verify justllm's logic without
network access or spending a cent on real calls.
"""
from __future__ import annotations

import pytest

from justllm import LLM, RetryPolicy, transports


# --- fake provider response objects (OpenAI-shaped) ---------------------------
class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Resp:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [type("C", (), {"message": _Msg(content, tool_calls)})()]


class _Err(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status_code = status


@pytest.fixture(autouse=True)
def _clear_exact_cache():
    transports._EXACT_CACHE.clear()
    yield
    transports._EXACT_CACHE.clear()


def _patch_completion(monkeypatch, fn):
    import litellm

    monkeypatch.setattr(litellm, "completion", fn)


# --- transport ----------------------------------------------------------------
def test_call_returns_text(monkeypatch):
    seen = {}

    def fake(model, messages, **kw):
        seen["model"] = model
        return _Resp(content="hello")

    _patch_completion(monkeypatch, fake)
    llm = LLM("openai/gpt-4o", compress=False)
    assert llm("hi") == "hello"
    assert seen["model"] == "openai/gpt-4o"


def test_anthropic_prompt_cache_breakpoint(monkeypatch):
    seen = {}

    def fake(model, messages, **kw):
        seen["messages"] = messages
        return _Resp(content="x")

    _patch_completion(monkeypatch, fake)
    LLM("anthropic/claude-opus-4-8", compress=False, cache="prompt")("hello")
    content = seen["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_openai_prompt_cache_is_noop(monkeypatch):
    seen = {}

    def fake(model, messages, **kw):
        seen["messages"] = messages
        return _Resp(content="x")

    _patch_completion(monkeypatch, fake)
    LLM("openai/gpt-4o", compress=False, cache="prompt")("hello")
    # OpenAI caches automatically; content stays a plain string.
    assert seen["messages"][0]["content"] == "hello"


def test_exact_cache_serves_second_call(monkeypatch):
    n = {"calls": 0}

    def fake(model, messages, **kw):
        n["calls"] += 1
        return _Resp(content="v")

    _patch_completion(monkeypatch, fake)
    llm = LLM("openai/gpt-4o", compress=False, cache="exact")
    assert llm("same question") == "v"
    assert llm("same question") == "v"
    assert n["calls"] == 1  # second served from cache


# --- fallback integration -----------------------------------------------------
def test_fallback_to_second_model(monkeypatch):
    def fake(model, messages, **kw):
        if model == "openai/bad":
            raise _Err(503)
        return _Resp(content="ok")

    _patch_completion(monkeypatch, fake)
    llm = LLM(
        chain=["openai/bad", "openai/good"],
        compress=False,
        retry=RetryPolicy(max_attempts=1),
    )
    assert llm("hi") == "ok"


# --- structured output --------------------------------------------------------
def test_extract_returns_validated_instance(monkeypatch):
    import instructor
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, response_model, **kw):
                    return response_model(name="Ada")

    monkeypatch.setattr(instructor, "from_litellm", lambda *a, **k: _FakeClient())
    person = LLM("openai/gpt-4o", compress=False).extract(Person, "who?")
    assert person.name == "Ada"


# --- agent loop ---------------------------------------------------------------
def test_agent_runs_tool_then_answers(monkeypatch):
    responses = iter(
        [
            _Resp(tool_calls=[_ToolCall("c1", "get_weather", '{"city": "Boston"}')]),
            _Resp(content="Pack a raincoat."),
        ]
    )

    def fake(model, messages, **kw):
        return next(responses)

    _patch_completion(monkeypatch, fake)
    agent = LLM("openai/gpt-4o", compress=False).agent(system="travel")

    calls = {}

    @agent.tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        calls["city"] = city
        return "rainy"

    out = agent.run("what to pack for Boston?")
    assert out == "Pack a raincoat."
    assert calls["city"] == "Boston"


def test_agent_respects_max_steps(monkeypatch):
    def always_tool(model, messages, **kw):
        return _Resp(tool_calls=[_ToolCall("c1", "noop", "{}")])

    _patch_completion(monkeypatch, always_tool)
    agent = LLM("openai/gpt-4o", compress=False).agent(max_steps=2)

    @agent.tool
    def noop() -> str:
        """Does nothing."""
        return "ok"

    assert "max_steps" in agent.run("loop forever")
