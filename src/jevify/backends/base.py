"""The whole backend contract: given a chat prompt, what is the next-token distribution?"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class NextToken:
    """Next-token log-probabilities after the prompt (top-k or exact, backend-dependent)."""

    logprobs: dict[str, float]  # token text -> natural-log probability
    prompt_tokens: int | None = None
    cached_tokens: int | None = None  # prefix-cache hits, if the runtime reports them
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str
    model: str

    def next_token(self, messages: list[dict[str, str]]) -> NextToken: ...
