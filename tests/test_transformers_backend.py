"""Slow: downloads Qwen/Qwen2.5-0.5B-Instruct (~1 GB) and runs it in-process.

    uv run --extra transformers pytest -m slow tests/test_transformers_backend.py
"""
import pytest

torch = pytest.importorskip("torch")

from jevify import Choice, Jevify, Noul, Score  # noqa: E402
from jevify.backends.transformers_backend import TransformersBackend  # noqa: E402

pytestmark = pytest.mark.slow

STATE = ("Ticket history entry: unrelated question, resolved. " * 40) + "\n\nLATEST MESSAGE: Help! My payouts have been failing for 3 days."
QS = {
    "department": Choice("Which team should handle the LATEST MESSAGE?", {"billing": "Payments, refunds", "technical": "Bugs, outages", "sales": "Pricing, upgrades"}),
    "is_urgent": Noul("Does the LATEST MESSAGE convey urgency?"),
    "frustration": Score("How frustrated is the customer in the LATEST MESSAGE?", ["Calm", "Frustrated", "Very angry"]),
}


@pytest.fixture(scope="module")
def backend():
    return TransformersBackend("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)


def _probs(ans):
    return ans["probabilities"] if "probabilities" in ans else {"yes": ans["noul"]}


def test_batched_and_packed_match_sequential(backend):
    jev = Jevify(backend)
    seq = {k: jev.system_one(STATE, {k: q}, debug=True)["answers"][k] for k, q in QS.items()}  # one question per call
    backend.packed = False
    branches = jev.system_one(STATE, QS, debug=True)
    backend.packed = True
    packed = jev.system_one(STATE, QS, debug=True)
    for k in QS:
        a = _probs(seq[k])
        for name, res in (("branches", branches), ("packed", packed)):
            b = _probs(res["answers"][k])
            for key in a:
                assert a[key] == pytest.approx(b[key], abs=2e-3), (name, k, a, b)
    assert branches["usage"]["cached_tokens"] > 0 and packed["usage"]["cached_tokens"] > 0


def test_answers_are_sane(backend):
    out = Jevify(backend).system_one(STATE, QS)
    dep = out["answers"]["department"]
    assert dep["choice"] in {"billing", "technical"}
    assert sum(dep["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["answers"]["is_urgent"]["noul"] > 0.5
    assert 0 <= out["answers"]["frustration"]["score"] <= 2
    assert out["usage"]["output_tokens"] == 0
