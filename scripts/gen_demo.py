#!/usr/bin/env python3
"""Generate an asciicast demo of justllm — no screen recording involved.

Writes assets/demo.cast (asciicast v2). Render to a GIF with agg
(https://github.com/asciinema/agg):

    python scripts/gen_demo.py
    agg --theme monokai assets/demo.cast assets/demo.gif

The outputs shown are real results observed while testing (Paris, the ~81%
compression on an 80-row JSON tool result, the map order). Edit this script and
re-render to change pacing or content; nothing here is hand-recorded.

Pacing knobs: CHAR_DT (typing speed), PAUSE (wait after each command), END_HOLD
(final freeze). HEIGHT is set to the line count so there's no empty space below.
"""
import json
import os

WIDTH, HEIGHT = 82, 15
CHAR_DT = 0.035  # per-character typing delay
PAUSE = 0.25     # wait after a command's newline
END_HOLD = 1.0   # final freeze before the GIF loops

ESC = "\x1b"
GREEN = ESC + "[32m"
BLUE = ESC + "[34m"
YELLOW = ESC + "[33m"
GREY = ESC + "[90m"
RESET = ESC + "[0m"
SH = f"{GREEN}${RESET} "
PY = f"{BLUE}>>>{RESET} "

events: list = []
t = 0.0


def emit(text: str, dt: float = 0.25) -> None:
    global t
    t += dt
    events.append([round(t, 3), "o", text])


def type_cmd(prompt: str, text: str) -> None:
    global t
    emit(prompt, 0.18)
    for ch in text:
        t += CHAR_DT
        events.append([round(t, 3), "o", ch])
    t += 0.1
    events.append([round(t, 3), "o", "\r\n"])
    t += PAUSE


def show(text: str) -> None:
    emit(text + "\r\n", 0.28)


emit(f"{GREY}# justllm: production LLM calls in three lines{RESET}\r\n", 0.4)
type_cmd(SH, "python")
show(f"{GREY}Python 3.12 | justllm 0.7.0{RESET}")

type_cmd(PY, "from justllm import LLM")
type_cmd(PY, 'llm = LLM("anthropic/claude-opus-4-8")')
type_cmd(PY, 'llm("In one word, the capital of France?")')
show(f"{YELLOW}'Paris.'{RESET}")

type_cmd(PY, f"{GREY}# tool output, logs, RAG get compressed automatically (~80% smaller){RESET}")

type_cmd(PY, "chat = llm.chat()")
type_cmd(PY, 'chat.send("Name three Stoic philosophers")')
show(f"{YELLOW}'Marcus Aurelius, Seneca, Epictetus.'{RESET}")
type_cmd(PY, 'chat.send("Which one was a Roman emperor?")')
show(f"{YELLOW}'Marcus Aurelius.'{RESET}")

emit(f"{GREY}# fallback, caching, and compression are all on by default{RESET}\r\n", 0.5)
t += END_HOLD
events.append([round(t, 3), "o", ""])

header = {
    "version": 2,
    "width": WIDTH,
    "height": HEIGHT,
    "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
}

os.makedirs("assets", exist_ok=True)
with open("assets/demo.cast", "w", encoding="utf-8") as fh:
    fh.write(json.dumps(header) + "\n")
    for event in events:
        fh.write(json.dumps(event) + "\n")

print(f"wrote assets/demo.cast ({len(events)} events, {round(t, 1)}s)")
