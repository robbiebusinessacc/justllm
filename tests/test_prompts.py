"""Tests for the prompt-loader seam."""
from __future__ import annotations

import pytest

from justllm import prompts


def test_load_from_file_and_render(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "greet.txt").write_text("Hello {name}, welcome to {place}.")
    prompts.set_base_dir(str(d))
    try:
        out = prompts.load("greet", name="Robbie", place="justllm")
        assert out == "Hello Robbie, welcome to justllm."
    finally:
        prompts.set_base_dir("prompts")


def test_literal_braces_are_preserved(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "t.md").write_text('Reply as JSON: {"key": "{val}"}')
    prompts.set_base_dir(str(d))
    try:
        # Only {val} is substituted; the JSON braces stay literal.
        assert prompts.load("t", val="V") == 'Reply as JSON: {"key": "V"}'
    finally:
        prompts.set_base_dir("prompts")


def test_missing_prompt_raises(tmp_path):
    prompts.set_base_dir(str(tmp_path))
    try:
        with pytest.raises(FileNotFoundError):
            prompts.load("does-not-exist")
    finally:
        prompts.set_base_dir("prompts")


def test_custom_loader_overrides_files():
    prompts.set_loader(lambda name: f"[{name}] {{x}}")
    try:
        assert prompts.load("anything", x="ok") == "[anything] ok"
    finally:
        prompts.set_loader(None)


def test_file_loader_caches_then_hot_reloads(tmp_path):
    import os

    f = tmp_path / "p.txt"
    f.write_text("v1 {x}")
    reloads = []
    prompts.set_loader(
        prompts.file_loader(str(tmp_path), on_reload=lambda name, path: reloads.append(name))
    )
    try:
        assert prompts.load("p", x="1") == "v1 1"
        assert prompts.load("p", x="2") == "v1 2"  # served from cache
        assert reloads == ["p"]  # read from disk only once

        f.write_text("v2 {x}")
        os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 10))  # force newer mtime
        assert prompts.load("p", x="3") == "v2 3"  # picked up the edit
        assert reloads == ["p", "p"]  # reloaded exactly once more
    finally:
        prompts.set_loader(None)


def test_file_loader_no_cache_always_rereads(tmp_path):
    (tmp_path / "p.txt").write_text("a")
    reloads = []
    prompts.set_loader(
        prompts.file_loader(
            str(tmp_path), cache=False, on_reload=lambda name, path: reloads.append(name)
        )
    )
    try:
        prompts.load("p")
        prompts.load("p")
        assert reloads == ["p", "p"]
    finally:
        prompts.set_loader(None)


def test_file_loader_missing_raises(tmp_path):
    prompts.set_loader(prompts.file_loader(str(tmp_path)))
    try:
        with pytest.raises(FileNotFoundError):
            prompts.load("nope")
    finally:
        prompts.set_loader(None)


def test_langfuse_loader_with_injected_client():
    class _FakePrompt:
        def get_langchain_prompt(self):
            return "Summarize {document} in {n} words."

    class _FakeClient:
        def get_prompt(self, name, label=None, version=None):
            assert name == "summary"
            return _FakePrompt()

    prompts.set_loader(prompts.langfuse_loader(client=_FakeClient()))
    try:
        out = prompts.load("summary", document="X", n="5")
        assert out == "Summarize X in 5 words."
    finally:
        prompts.set_loader(None)


def test_langfuse_loader_rejects_chat_prompt():
    class _ChatPrompt:
        def get_langchain_prompt(self):
            return [{"role": "system", "content": "hi"}]  # not a string

    class _FakeClient:
        def get_prompt(self, name, label=None, version=None):
            return _ChatPrompt()

    prompts.set_loader(prompts.langfuse_loader(client=_FakeClient()))
    try:
        with pytest.raises(TypeError):
            prompts.load("chatty")
    finally:
        prompts.set_loader(None)
