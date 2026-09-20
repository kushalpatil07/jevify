"""Render a (state, question) pair into chat messages whose *first assistant token* is the answer.

Design notes (from spike/ results):
- Every prompt for one `system_one` call starts with the identical `<state>` block so the
  runtime's prompt cache (llama.cpp/Ollama `cache_prompt`, vLLM prefix caching) turns the
  2nd..Nth question into a few-hundred-token prefill.
- Labels are single tokens in every tokenizer we've met: letters for Choice *and* Score
  (digit labels made Qwen3 answer "1" for everything), "Yes"/"No" for Noul.
- No assistant prefill: with an explicit system instruction the models put ~100% of the
  next-token mass on the labels; an "Answer: " prefill actually *hurt* (models continued
  with digits from the state).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .images import is_image_ref
from .questions import Choice, Noul, Question, Score

SYSTEM_PROMPT = "You are a precise classifier. Reply with a single token: only the label of your answer."

LETTERS = [chr(ord("A") + i) for i in range(26)]


@dataclass(frozen=True)
class Rendered:
    messages: list[dict[str, Any]]  # user content is a str, or a list of image/text blocks when the state has images
    labels: list[str]  # what the model should emit, e.g. ["A", "B", "C"]
    keys: list[str]  # what each label means to the user, same order


def split_state(state: Any) -> tuple[Any, list[Any]]:
    """Pull image references out of the state. Images can be the whole state, or live under
    an "image" / "images" key of a dict; the rest of the dict stays as text."""
    if is_image_ref(state):
        return None, [state]
    if isinstance(state, list) and state and all(is_image_ref(x) for x in state):
        return None, list(state)
    if isinstance(state, dict):
        images = []
        rest = {}
        for k, v in state.items():
            if k in ("image", "images") and (is_image_ref(v) or (isinstance(v, list) and v and all(is_image_ref(x) for x in v))):
                images += v if isinstance(v, list) else [v]
            else:
                rest[k] = v
        return (rest or None), images
    return state, []


def render_state(state: Any, n_images: int = 0) -> str:
    parts = [f"[image {i + 1} attached]" for i in range(n_images)] if n_images else []
    if state is not None:
        parts.append(state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False))
    return "<state>\n" + "\n".join(parts) + "\n</state>\n\n"


def render(state: Any, question: Question, system_prompt: str = SYSTEM_PROMPT) -> Rendered:
    text_state, images = split_state(state)
    head = render_state(text_state, len(images)) + question.instructions.strip() + "\n\n"

    if isinstance(question, Choice):
        keys = list(question.criteria)
        labels = LETTERS[: len(keys)]
        lines = [f"{lab}. {k}" + (f" — {desc}" if desc else "") for lab, (k, desc) in zip(labels, question.criteria.items())]
        body = "Options:\n" + "\n".join(lines) + "\n\nAnswer with the letter of the single best option."
    elif isinstance(question, Score):
        keys = [str(i) for i in range(len(question.criteria))]
        labels = LETTERS[: len(keys)]
        lines = [f"{lab}. {lvl}" for lab, lvl in zip(labels, question.criteria)]
        body = "Levels (ordered from lowest to highest):\n" + "\n".join(lines) + "\n\nAnswer with the letter of the single best level."
    elif isinstance(question, Noul):
        keys = ["yes", "no"]
        labels = ["Yes", "No"]
        body = ""
        if question.criteria:
            if question.criteria.get("true"):
                body += f"Yes if: {question.criteria['true']}\n"
            if question.criteria.get("false"):
                body += f"No if: {question.criteria['false']}\n"
            body += "\n"
        body += "Answer Yes or No."
    else:  # pragma: no cover
        raise TypeError(f"not a question: {question!r}")

    user: Any = head + body
    if images:  # image blocks first, then the text; backends turn these into their own format
        user = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": head + body}]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    return Rendered(messages=messages, labels=labels, keys=keys)
