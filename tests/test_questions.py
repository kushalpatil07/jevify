import pytest

from jevify.questions import Choice, Noul, Score, question_from_dict, question_to_dict


def test_parse_jev_docs_request():
    qs = {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?", "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
        "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
    }
    parsed = {k: question_from_dict(v) for k, v in qs.items()}
    assert isinstance(parsed["is_urgent"], Noul)
    assert isinstance(parsed["department"], Choice)
    assert isinstance(parsed["frustration"], Score)
    assert parsed["frustration"].criteria == ("Calm", "Frustrated", "Very angry")
    # round-trips
    for k, v in qs.items():
        assert question_to_dict(parsed[k]) == v


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "nope", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": {"only": None}},
        {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
        {"type": "score", "instructions": "x", "criteria": ["one"]},
        {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
        {"type": "noul", "instructions": ""},
        {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}},
        "not a dict",
    ],
)
def test_rejects_invalid(bad):
    with pytest.raises(ValueError):
        question_from_dict(bad)
