"""`Jevify.system_one(state, questions)` -> Jev-shaped response dict."""
from __future__ import annotations

from typing import Any, Mapping

from .backends.base import Backend
from .backends.openai_compat import OpenAICompatBackend
from .prompting import SYSTEM_PROMPT, render
from .questions import Question, question_from_dict
from .scoring import answer_from_logprobs


class Jevify:
    def __init__(self, backend: Backend, system_prompt: str = SYSTEM_PROMPT):
        self.backend = backend
        self.system_prompt = system_prompt

    # -- constructors -----------------------------------------------------------------
    @classmethod
    def from_openai(cls, model: str, base_url: str | None = None, runtime: str = "openai", **kw) -> "Jevify":
        return cls(OpenAICompatBackend(model=model, base_url=base_url, runtime=runtime, **kw))

    @classmethod
    def from_runtime(cls, runtime: str, model: str, **kw) -> "Jevify":
        """`Jevify.from_runtime("ollama", "gemma4:e4b")`"""
        return cls.from_openai(model=model, runtime=runtime, **kw)

    @classmethod
    def from_transformers(cls, model: str, **kw) -> "Jevify":
        from .backends.transformers_backend import TransformersBackend

        return cls(TransformersBackend(model, **kw))

    # -- the one call -------------------------------------------------------------------
    def system_one(self, state: Any, questions: Mapping[str, Question | dict], debug: bool = False) -> dict:
        parsed: dict[str, Question] = {
            k: (q if not isinstance(q, dict) else question_from_dict(q)) for k, q in questions.items()
        }
        if not parsed:
            raise ValueError("questions must not be empty")

        rendered = {key: render(state, q, self.system_prompt) for key, q in parsed.items()}

        # Backends that can evaluate all branches in one forward pass expose `next_token_batch`;
        # otherwise ask sequentially (the runtime's prompt cache still shares the state prefill).
        batch_fn = getattr(self.backend, "next_token_batch", None)
        if batch_fn is not None:
            results = batch_fn([r.messages for r in rendered.values()])
        else:
            results = [self.backend.next_token(r.messages) for r in rendered.values()]

        answers: dict[str, dict] = {}
        input_tokens = 0
        cached_tokens = 0
        for (key, q), r, nt in zip(parsed.items(), rendered.values(), results):
            answers[key] = answer_from_logprobs(q, r.labels, r.keys, nt.logprobs, debug=debug)
            input_tokens += nt.prompt_tokens or 0
            cached_tokens += nt.cached_tokens or 0

        out = {
            "model": self.backend.model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
        if debug:
            out["usage"]["cached_tokens"] = cached_tokens
            out["backend"] = self.backend.name
        return out
