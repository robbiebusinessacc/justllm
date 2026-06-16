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
