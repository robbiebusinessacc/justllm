"""Every cookbook example must stay syntactically valid as the API evolves.

This compiles (does not execute) each file in examples/, so the cookbook can't
silently rot when the public API changes.
"""
from __future__ import annotations

import pathlib
import py_compile

import pytest

_EXAMPLES = sorted(
    (pathlib.Path(__file__).resolve().parent.parent / "examples").glob("*.py")
)


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.name)
def test_example_compiles(path):
    py_compile.compile(str(path), doraise=True)
