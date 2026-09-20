from jevify.prompting import SYSTEM_PROMPT, render
from jevify.questions import Choice, Noul, Score

STATE = "Help! My payouts have been failing for 3 days."


def test_choice_render():
    r = render(STATE, Choice("Which team?", {"billing": "Payments", "technical": None}))
    assert r.labels == ["A", "B"]
    assert r.keys == ["billing", "technical"]
    user = r.messages[1]["content"]
    assert user.startswith(f"<state>\n{STATE}\n</state>\n\n")
    assert "A. billing — Payments\nB. technical\n" in user
    assert r.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_score_uses_letters_and_index_keys():
    r = render(STATE, Score("How bad?", ["low", "mid", "high"]))
    assert r.labels == ["A", "B", "C"]
    assert r.keys == ["0", "1", "2"]
    assert "lowest to highest" in r.messages[1]["content"]


def test_noul_with_criteria():
    r = render(STATE, Noul("Urgent?", {"true": "time-sensitive", "false": "relaxed"}))
    assert r.labels == ["Yes", "No"] and r.keys == ["yes", "no"]
    assert "Yes if: time-sensitive" in r.messages[1]["content"]
    assert r.messages[1]["content"].endswith("Answer Yes or No.")


def test_shared_state_prefix_across_questions():
    qs = [Choice("q1", {"a": None, "b": None}), Noul("q2"), Score("q3", ["x", "y"])]
    users = [render(STATE, q).messages[1]["content"] for q in qs]
    prefix = f"<state>\n{STATE}\n</state>\n\n"
    assert all(u.startswith(prefix) for u in users)


def test_non_string_state_is_json():
    r = render({"ticket": 42, "text": "héllo"}, Noul("q"))
    assert '"ticket": 42' in r.messages[1]["content"]
    assert "héllo" in r.messages[1]["content"]  # ensure_ascii=False


def test_supports_packed_detects_sliding_layers():
    from types import SimpleNamespace

    from jevify.backends.transformers_backend import supports_packed

    assert supports_packed(SimpleNamespace(layer_types=None, sliding_window=None))
    assert supports_packed(SimpleNamespace(layer_types=["full_attention"] * 4, sliding_window=None))
    assert not supports_packed(SimpleNamespace(layer_types=["sliding_attention", "full_attention"], sliding_window=1024))


def test_image_state_becomes_image_blocks():
    r = render({"image": "https://example.com/a.png", "ticket": 7}, Noul("Is there a dog?"))
    user = r.messages[1]["content"]
    assert isinstance(user, list)
    assert user[0] == {"type": "image", "image": "https://example.com/a.png"}
    assert user[1]["type"] == "text"
    assert user[1]["text"].startswith("<state>\n[image 1 attached]\n{\n  \"ticket\": 7\n}\n</state>")
    # a bare URL / data URL as the whole state also works
    r2 = render("data:image/png;base64,AAAA", Noul("q"))
    assert r2.messages[1]["content"][0]["type"] == "image"
    # plain text state is untouched
    assert isinstance(render("hello", Noul("q")).messages[1]["content"], str)
