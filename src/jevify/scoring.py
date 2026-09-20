"""Turn next-token log-probabilities into Jev-shaped answers. Pure functions, no I/O."""
from __future__ import annotations

import math
from typing import Mapping

from .questions import Choice, Noul, Question, Score

ROUND = 4


def match_label_probs(token_logprobs: Mapping[str, float], labels: list[str]) -> tuple[dict[str, float], float]:
    """Sum the probability of every token whose stripped, lowercased text equals a label.

    Handles the usual tokenizer variants ("A", " A", "a", "\\tA"). Returns
    (unnormalized prob per label, total mass on labels).
    """
    wanted = {lab.lower(): lab for lab in labels}
    raw = {lab: 0.0 for lab in labels}
    for tok, lp in token_logprobs.items():
        key = tok.strip().lower()
        if key in wanted:
            raw[wanted[key]] += math.exp(lp)
    return raw, sum(raw.values())


def normalize(raw: Mapping[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        # No label token made it into the returned top-k: fall back to uniform so the
        # answer is still well-formed; callers can inspect `mass` to detect this.
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}


def confidence(probs: Mapping[str, float]) -> float:
    """1 - normalized entropy: 1.0 = all mass on one outcome, 0.0 = uniform.

    Jev does not publish its confidence formula; this is ours and is documented as such.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def _r(x: float) -> float:
    return round(x, ROUND)


def answer_from_logprobs(
    question: Question,
    labels: list[str],
    keys: list[str],
    token_logprobs: Mapping[str, float],
    debug: bool = False,
) -> dict:
    """Build one Jev answer object.

    `labels` are what the model was asked to emit (e.g. ["A","B","C"] or ["Yes","No"]);
    `keys` are the user-facing names they map to (option keys, "0".."N-1" for score,
    ["yes","no"] for noul), in the same order.
    """
    raw, mass = match_label_probs(token_logprobs, labels)
    probs_by_label = normalize(raw)
    probs = {k: probs_by_label[lab] for lab, k in zip(labels, keys)}

    if isinstance(question, Noul):
        out = {"type": "noul", "noul": _r(probs["yes"])}
    elif isinstance(question, Choice):
        best = max(probs, key=probs.get)
        out = {
            "type": "choice",
            "choice": best,
            "probabilities": {k: _r(p) for k, p in probs.items()},
            "confidence": _r(confidence(probs)),
        }
    elif isinstance(question, Score):
        expected = sum(int(k) * p for k, p in probs.items())
        out = {
            "type": "score",
            "score": _r(expected),
            "legend": {str(i): lvl for i, lvl in enumerate(question.criteria)},
            "probabilities": {k: _r(p) for k, p in probs.items()},
            "confidence": _r(confidence(probs)),
        }
    else:  # pragma: no cover
        raise TypeError(f"not a question: {question!r}")

    if debug:
        out["debug"] = {
            "label_mass": _r(mass),
            "top_tokens": sorted(((t, _r(math.exp(lp))) for t, lp in token_logprobs.items()), key=lambda kv: -kv[1])[:8],
        }
    return out
