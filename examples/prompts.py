"""Prompts: load templates from files, hot-reload them, or pull from a registry.

No API call; this just shows the loader seam. Only the variables you pass get
substituted, so literal braces (JSON, code) survive.
"""
import os
import tempfile

from justllm import prompts


def main() -> None:
    # A prompt living on disk.
    base = tempfile.mkdtemp()
    with open(os.path.join(base, "summarize.txt"), "w") as fh:
        fh.write("Summarize the following in {n} words:\n\n{text}")

    # 1. default loader: load + render.
    prompts.set_base_dir(base)
    print(prompts.load("summarize", n="5", text="justllm is a tiny LLM front door."))

    # 2. hot-reloading loader: caches, re-reads only when the file's mtime changes.
    prompts.set_loader(prompts.file_loader(base))
    print(prompts.load("summarize", n="3", text="Edits are picked up automatically."))

    # 3. registry adapter (needs `pip install 'justllm[langfuse]'` + credentials):
    #    prompts.set_loader(prompts.langfuse_loader(label="production"))

    prompts.set_loader(None)  # back to the default file loader


if __name__ == "__main__":
    main()
