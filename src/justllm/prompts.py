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


def _resolve(name: str, base_dir: Optional[str] = None) -> Optional[str]:
    base = base_dir if base_dir is not None else _BASE_DIR
    candidates = [name] + [os.path.join(base, name + ext) for ext in _EXTENSIONS]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _not_found(name: str, base_dir: Optional[str] = None) -> FileNotFoundError:
    base = base_dir if base_dir is not None else _BASE_DIR
    exts = ",".join(e for e in _EXTENSIONS if e)
    return FileNotFoundError(
        f"prompt {name!r} not found (looked for {name} and {base}/{name}{{{exts}}})"
    )


def _read_file(name: str) -> str:
    path = _resolve(name)
    if path is None:
        raise _not_found(name)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


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


def file_loader(
    base_dir: Optional[str] = None,
    *,
    cache: bool = True,
    on_reload: Optional[Callable[[str, str], None]] = None,
) -> Callable[[str], str]:
    """A file loader for ``set_loader`` with mtime-based hot-reload.

    Caches each prompt by path and re-reads only when the file's modification
    time changes — edits are picked up without a restart, while unchanged prompts
    are served from memory (the plain default loader re-reads on every call).
    Pass ``cache=False`` to always re-read, and ``on_reload(name, path)`` to
    observe (re)loads.

        prompts.set_loader(prompts.file_loader("prompts"))
    """
    store: dict = {}  # path -> (mtime, text)

    def loader(name: str) -> str:
        path = _resolve(name, base_dir)
        if path is None:
            raise _not_found(name, base_dir)
        mtime = os.path.getmtime(path)
        if cache:
            cached = store.get(path)
            if cached is not None and cached[0] == mtime:
                return cached[1]
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if cache:
            store[path] = (mtime, text)
        if on_reload is not None:
            on_reload(name, path)
        return text

    return loader


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
