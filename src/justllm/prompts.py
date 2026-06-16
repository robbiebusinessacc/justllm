"""A tiny prompt-loader seam.

Load prompts from files and render them with variables — without building a
prompt registry (that's a hosted-service concern). Swap in Langfuse, a database,
or anything else with ``set_loader()``.

    # prompts/greeting.txt  ->  "Hello {name}, welcome to {place}."
    from justllm import prompts
    prompts.load("greeting", name="Robbie", place="justllm")

Rendering only substitutes the variables you pass (``{name}``), so templates can
contain other literal braces (JSON, code) without escaping.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

_BASE_DIR = "prompts"
_LOADER: Optional[Callable[[str], str]] = None
_EXTENSIONS = ("", ".prompt", ".txt", ".md", ".jinja", ".j2")


def set_base_dir(path: str) -> None:
    """Set the directory the default file loader searches."""
    global _BASE_DIR
    _BASE_DIR = path


def set_loader(loader: Optional[Callable[[str], str]]) -> None:
    """Install a custom template source: ``loader(name) -> template string``.

    Pass ``None`` to revert to the default file loader.
    """
    global _LOADER
    _LOADER = loader


def _read_file(name: str) -> str:
    candidates = [name] + [
        os.path.join(_BASE_DIR, name + ext) for ext in _EXTENSIONS
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read()
    raise FileNotFoundError(
        f"prompt {name!r} not found (looked for {name} and "
        f"{_BASE_DIR}/{name}{{{','.join(e for e in _EXTENSIONS if e)}}})"
    )


def _render(template: str, variables: dict) -> str:
    out = template
    for key, value in variables.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def load(name: str, /, **variables) -> str:
    """Load a prompt template by name and render it with ``variables``.

    ``name`` is positional-only so a template variable can also be called
    ``name`` (e.g. ``load("greet", name="Robbie")``).
    """
    template = _LOADER(name) if _LOADER else _read_file(name)
    return _render(template, variables) if variables else template


def langfuse_loader(
    *,
    label: Optional[str] = None,
    version: Optional[int] = None,
    client: Any = None,
    **client_kwargs: Any,
) -> Callable[[str], str]:
    """A loader that fetches text prompts from Langfuse, for ``set_loader``.

        from justllm import prompts
        prompts.set_loader(prompts.langfuse_loader(label="production"))
        prompts.load("summary", document=text)   # fetched from Langfuse, then rendered

    Returns Langfuse's LangChain-style template (``{{var}}`` -> ``{var}``), which
    this module's renderer then fills — so it composes with the existing seam, no
    special-casing. Needs ``pip install 'justllm[langfuse]'`` and Langfuse
    credentials (env vars, or pass them as ``client_kwargs``). Text prompts only.
    """
    state = {"client": client}

    def loader(name: str) -> str:
        if state["client"] is None:
            from langfuse import Langfuse

            state["client"] = Langfuse(**client_kwargs)
        prompt = state["client"].get_prompt(name, label=label, version=version)
        template = prompt.get_langchain_prompt()
        if not isinstance(template, str):
            raise TypeError(
                f"langfuse_loader supports text prompts; {name!r} is a chat prompt."
            )
        return template

    return loader
