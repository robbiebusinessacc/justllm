"""Wiring tests for the call path: transport, caching, fallback, streaming,
async, routing, structured output, and the agent loop.

LiteLLM/instructor are monkeypatched, so these verify justllm's logic without
network access or spending a cent on real calls.
"""
from __future__ import annotations

import asyncio

import pytest

from justllm import LLM, Cascade, RetryPolicy, Router, embedding_escalator, transports
from justllm import observability as obs


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


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _Resp:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [type("C", (), {"message": _Msg(content, tool_calls)})()]
        self.usage = _Usage()


class _Delta:
    def __init__(self, content):
        self.content = content


def _stream_chunks(pieces):
    for p in pieces:
        yield type("Chunk", (), {"choices": [type("C", (), {"delta": _Delta(p)})()]})()


class _Err(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status_code = status


def _has_headroom() -> bool:
    try:
        import headroom  # noqa: F401

        return True
    except Exception:
        return False


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
    assert LLM("openai/gpt-4o", compress=False)("hi") == "hello"
    assert seen["model"] == "openai/gpt-4o"


# --- cache --------------------------------------------------------------------
def test_prompt_cache_invokes_optimizer(monkeypatch):
    monkeypatch.setattr(
        transports, "_optimize_cache",
        lambda model, msgs: [{"role": "user", "content": "TAGGED"}],
    )
    seen = {}

    def fake(model, messages, **kw):
        seen["messages"] = messages
        return _Resp(content="x")

    _patch_completion(monkeypatch, fake)
    LLM("anthropic/claude-opus-4-8", compress=False, cache="prompt")("hello")
    assert seen["messages"][0]["content"] == "TAGGED"


def test_cache_off_skips_optimizer(monkeypatch):
    monkeypatch.setattr(
        transports, "_optimize_cache",
        lambda model, msgs: [{"role": "user", "content": "TAGGED"}],
    )
    seen = {}

    def fake(model, messages, **kw):
        seen["messages"] = messages
        return _Resp(content="x")

    _patch_completion(monkeypatch, fake)
    LLM("anthropic/claude-opus-4-8", compress=False, cache="off")("hello")
    assert seen["messages"][0]["content"] == "hello"


@pytest.mark.skipif(not _has_headroom(), reason="headroom-ai not installed")
def test_optimize_cache_inserts_anthropic_breakpoint():
    # Must exceed Headroom's 1024-token minimum for a cache breakpoint.
    messages = [
        {"role": "system", "content": "You are helpful.\n" + "Reference material. " * 600},
        {"role": "user", "content": "hi"},
    ]
    out = transports._optimize_cache("anthropic/claude-sonnet-4-5", messages)
    assert "cache_control" in repr(out)


def test_exact_cache_serves_second_call(monkeypatch):
    n = {"calls": 0}

    def fake(model, messages, **kw):
        n["calls"] += 1
        return _Resp(content="v")

    _patch_completion(monkeypatch, fake)
    llm = LLM("openai/gpt-4o", compress=False, cache="exact")
    assert llm("same question") == "v"
    assert llm("same question") == "v"
    assert n["calls"] == 1


# --- fallback -----------------------------------------------------------------
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


# --- streaming ----------------------------------------------------------------
def test_stream_yields_chunks(monkeypatch):
    def fake(model, messages, stream=False, **kw):
        assert stream is True
        return _stream_chunks(["Hel", "lo", "!"])

    _patch_completion(monkeypatch, fake)
    out = list(LLM("openai/gpt-4o", compress=False).stream("hi"))
    assert "".join(out) == "Hello!"


# --- async --------------------------------------------------------------------
def test_acall_returns_text(monkeypatch):
    import litellm

    async def fake_acompletion(model, messages, **kw):
        return _Resp(content="async-ok")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    out = asyncio.run(LLM("openai/gpt-4o", compress=False).acall("hi"))
    assert out == "async-ok"


# --- routing ------------------------------------------------------------------
def test_router_picks_by_length():
    r = Router(small="s", large="l", max_small_tokens=5)
    assert r.choose("hi") == "s"
    assert r.choose("word " * 80) == "l"


def test_router_used_by_call(monkeypatch):
    seen = {}

    def fake(model, messages, **kw):
        seen["model"] = model
        return _Resp(content="ok")

    _patch_completion(monkeypatch, fake)
    r = Router(small="openai/small", large="openai/large", max_small_tokens=5)
    LLM(router=r, compress=False, cache="off")("hi")
    assert seen["model"] == "openai/small"


def test_cascade_no_escalation(monkeypatch):
    calls = []

    def fake(model, messages, **kw):
        calls.append(model)
        return _Resp(content="A complete, confident answer.")

    _patch_completion(monkeypatch, fake)
    out = LLM(
        router=Cascade(small="m/small", large="m/large"), compress=False, cache="off"
    )("q")
    assert out == "A complete, confident answer."
    assert calls == ["m/small"]  # never touched the large model


def test_cascade_escalates_on_refusal(monkeypatch):
    def fake(model, messages, **kw):
        return _Resp(content="I don't know." if model == "m/small" else "Paris.")

    _patch_completion(monkeypatch, fake)
    out = LLM(
        router=Cascade(small="m/small", large="m/large"), compress=False, cache="off"
    )("q")
    assert out == "Paris."


def test_cascade_custom_predicate(monkeypatch):
    calls = []

    def fake(model, messages, **kw):
        calls.append(model)
        return _Resp(content="anything")

    _patch_completion(monkeypatch, fake)
    LLM(
        router=Cascade(small="m/small", large="m/large", escalate_if=lambda a: True),
        compress=False,
        cache="off",
    )("q")
    assert calls == ["m/small", "m/large"]  # predicate forces escalation


def _fake_embed(texts):
    # refusal/uncertainty words -> [1, 0]; everything else -> [0, 1]
    markers = ("know", "sure", "unable", "can't", "cannot", "unclear", "enough")
    return [
        [1.0, 0.0] if any(w in t.lower() for w in markers) else [0.0, 1.0]
        for t in texts
    ]


def test_embedding_escalator_detects_semantic_refusal():
    escalate = embedding_escalator(embed=_fake_embed, threshold=0.9)
    assert escalate("I really don't know, sorry.") is True  # near a refusal exemplar
    assert escalate("The capital is Paris.") is False  # a real answer
    assert escalate("") is True  # empty -> escalate


def test_embedding_escalator_in_cascade(monkeypatch):
    def fake(model, messages, **kw):
        return _Resp(content="I don't know." if model == "m/small" else "Paris.")

    _patch_completion(monkeypatch, fake)
    cascade = Cascade(
        small="m/small",
        large="m/large",
        escalate_if=embedding_escalator(embed=_fake_embed, threshold=0.9),
    )
    out = LLM(router=cascade, compress=False, cache="off")("q")
    assert out == "Paris."  # cheap answer was a refusal -> escalated


# --- observability ------------------------------------------------------------
def test_cost_of_uses_pricing_map():
    assert abs(obs.cost_of("gpt-4o", 1_000_000, 0) - 2.50) < 1e-6
    assert obs.cost_of("some-unknown-model", 1000, 1000) is None


def test_call_span_is_noop_without_otel():
    with obs.call_span("openai/gpt-4o") as rec:
        rec.record(_Resp(content="x"))  # must not raise


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


def test_aextract_returns_validated_instance(monkeypatch):
    import instructor
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(model, messages, response_model, **kw):
                    return response_model(name="Zed")

    monkeypatch.setattr(instructor, "from_litellm", lambda *a, **k: _FakeClient())
    person = asyncio.run(LLM("openai/gpt-4o", compress=False).aextract(Person, "who?"))
    assert person.name == "Zed"


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
