import math

import pytest

from jevify.questions import Choice, Noul, Score
from jevify.scoring import answer_from_logprobs, confidence, match_label_probs


def lp(**probs):
    return {k: math.log(v) for k, v in probs.items()}


def test_label_matching_merges_token_variants():
    raw, mass = match_label_probs({"A": math.log(0.5), " A": math.log(0.2), "a": math.log(0.1), "B": math.log(0.1), "The": math.log(0.1)}, ["A", "B"])
    assert raw["A"] == pytest.approx(0.8)
    assert raw["B"] == pytest.approx(0.1)
    assert mass == pytest.approx(0.9)


def test_choice_answer():
    q = Choice("which?", {"billing": None, "technical": None, "sales": None})
    out = answer_from_logprobs(q, ["A", "B", "C"], ["billing", "technical", "sales"], lp(A=0.08, B=0.85, C=0.07))
    assert out["type"] == "choice" and out["choice"] == "technical"
    assert out["probabilities"] == {"billing": 0.08, "technical": 0.85, "sales": 0.07}
    assert 0 < out["confidence"] < 1
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_choice_renormalizes_over_labels_only():
    q = Choice("which?", {"x": None, "y": None})
    out = answer_from_logprobs(q, ["A", "B"], ["x", "y"], lp(A=0.3, B=0.1, The=0.6), debug=True)
    assert out["probabilities"] == {"x": 0.75, "y": 0.25}
    assert out["debug"]["label_mass"] == pytest.approx(0.4)


def test_score_expectation_and_legend():
    q = Score("how?", ["Calm", "Frustrated", "Very angry"])
    out = answer_from_logprobs(q, ["A", "B", "C"], ["0", "1", "2"], lp(A=0.05, B=0.3, C=0.65))
    assert out["score"] == pytest.approx(0.3 + 1.3, abs=1e-3)  # 0*.05 + 1*.3 + 2*.65 = 1.6
    assert out["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert list(out["probabilities"]) == ["0", "1", "2"]


def test_noul():
    out = answer_from_logprobs(Noul("urgent?"), ["Yes", "No"], ["yes", "no"], lp(Yes=0.92, No=0.08))
    assert out == {"type": "noul", "noul": 0.92}


def test_missing_labels_fall_back_to_uniform():
    q = Choice("which?", {"x": None, "y": None})
    out = answer_from_logprobs(q, ["A", "B"], ["x", "y"], lp(The=1.0), debug=True)
    assert out["probabilities"] == {"x": 0.5, "y": 0.5}
    assert out["confidence"] == 0.0
    assert out["debug"]["label_mass"] == 0.0


def test_confidence_bounds():
    assert confidence({"a": 1.0, "b": 0.0}) == 1.0
    assert confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
    assert confidence({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == pytest.approx(0.0)
