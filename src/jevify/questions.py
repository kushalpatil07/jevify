"""Jev's three question primitives, as plain dataclasses.

Mirrors the wire format of https://docs.typesafe.ai/api :
  {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}}   # criteria optional
  {"type": "choice", "instructions": "...", "criteria": {"key": "description or null", ...}}
  {"type": "score",  "instructions": "...", "criteria": ["lowest level", ..., "highest level"]}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

MAX_CHOICE_OPTIONS = 26  # A..Z labels
MAX_SCORE_LEVELS = 10  # Jev's limit; also keeps labels single-token


@dataclass(frozen=True)
class Noul:
    """Binary judgment -> P(yes)."""

    instructions: str
    criteria: dict[str, str | None] | None = None  # optional {"true": ..., "false": ...}

    def __post_init__(self):
        if not self.instructions or not self.instructions.strip():
            raise ValueError("noul: instructions must be a non-empty string")
        if self.criteria is not None:
            extra = set(self.criteria) - {"true", "false"}
            if extra:
                raise ValueError(f"noul: criteria keys must be 'true'/'false', got {sorted(extra)}")


@dataclass(frozen=True)
class Choice:
    """Categorical classification -> distribution over `criteria` keys."""

    instructions: str
    criteria: dict[str, str | None] = field(default_factory=dict)

    def __post_init__(self):
        if not self.instructions or not self.instructions.strip():
            raise ValueError("choice: instructions must be a non-empty string")
        n = len(self.criteria)
        if not 2 <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice: needs 2..{MAX_CHOICE_OPTIONS} options, got {n}")
        if any(not k or not str(k).strip() for k in self.criteria):
            raise ValueError("choice: option keys must be non-empty strings")


@dataclass(frozen=True)
class Score:
    """Ordinal classification -> distribution over ordered levels + expected value."""

    instructions: str
    criteria: tuple[str, ...] = ()

    def __init__(self, instructions: str, criteria):
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "criteria", tuple(criteria or ()))
        if not instructions or not instructions.strip():
            raise ValueError("score: instructions must be a non-empty string")
        n = len(self.criteria)
        if not 2 <= n <= MAX_SCORE_LEVELS:
            raise ValueError(f"score: needs 2..{MAX_SCORE_LEVELS} ordered levels, got {n}")


Question = Union[Noul, Choice, Score]


def question_from_dict(d: Any) -> Question:
    """Parse one question from Jev's JSON shape."""
    if not isinstance(d, dict):
        raise ValueError(f"question must be an object, got {type(d).__name__}")
    qtype = d.get("type")
    instructions = d.get("instructions", "")
    criteria = d.get("criteria")
    if qtype == "noul":
        return Noul(instructions=instructions, criteria=criteria)
    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice: criteria must be an object of {option: description|null}")
        return Choice(instructions=instructions, criteria=dict(criteria))
    if qtype == "score":
        if not isinstance(criteria, list):
            raise ValueError("score: criteria must be an ordered list of level descriptions")
        return Score(instructions=instructions, criteria=criteria)
    raise ValueError(f"unknown question type {qtype!r}; expected 'noul', 'choice' or 'score'")


def question_to_dict(q: Question) -> dict:
    if isinstance(q, Noul):
        out = {"type": "noul", "instructions": q.instructions}
        if q.criteria is not None:
            out["criteria"] = dict(q.criteria)
        return out
    if isinstance(q, Choice):
        return {"type": "choice", "instructions": q.instructions, "criteria": dict(q.criteria)}
    if isinstance(q, Score):
        return {"type": "score", "instructions": q.instructions, "criteria": list(q.criteria)}
    raise TypeError(f"not a question: {q!r}")
