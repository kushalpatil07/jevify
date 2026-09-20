"""Backend + engine + server against a fake OpenAI-compatible endpoint (no model needed)."""
import json
import math

import httpx
from fastapi.testclient import TestClient

from jevify import Choice, Jevify, Noul, Score
from jevify.backends.openai_compat import OpenAICompatBackend
from jevify.server import create_app


def fake_server(answer_by_label: dict[str, float]):
    """Return an httpx transport that answers any chat/completions with a fixed top_logprobs list."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert body["max_tokens"] == 1 and body["logprobs"] is True and body["top_logprobs"] <= 20
        top = [{"token": t, "logprob": math.log(p)} for t, p in answer_by_label.items()]
        first = max(top, key=lambda t: t["logprob"])
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"role": "assistant", "content": first["token"]}, "logprobs": {"content": [{**first, "top_logprobs": top}]}}],
                "usage": {"prompt_tokens": 123, "prompt_tokens_details": {"cached_tokens": 100}},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, seen


def test_backend_parses_logprobs_and_usage():
    client, seen = fake_server({"B": 0.85, "A": 0.08, "C": 0.07})
    be = OpenAICompatBackend(model="m", runtime="ollama", client=client)
    nt = be.next_token([{"role": "user", "content": "hi"}])
    assert nt.logprobs["B"] == math.log(0.85)
    assert nt.prompt_tokens == 123 and nt.cached_tokens == 100
    assert seen[0]["reasoning_effort"] == "none"  # ollama no-think preset


def test_engine_end_to_end_shape():
    client, _ = fake_server({"B": 0.85, "A": 0.08, "C": 0.07, "Yes": 0.6, "No": 0.4})
    jev = Jevify(OpenAICompatBackend(model="m", runtime="ollama", client=client))
    out = jev.system_one(
        "Help!",
        {
            "department": Choice("Which team?", {"billing": None, "technical": None, "sales": None}),
            "frustration": Score("How mad?", ["Calm", "Frustrated", "Very angry"]),
            "is_urgent": {"type": "noul", "instructions": "Urgent?"},
        },
    )
    assert set(out) == {"model", "answers", "usage"}
    assert out["answers"]["department"]["choice"] == "technical"
    assert out["answers"]["frustration"]["score"] == 0.99  # 0*.08 + 1*.85 + 2*.07
    assert out["answers"]["is_urgent"] == {"type": "noul", "noul": 0.6}
    assert out["usage"] == {"input_tokens": 3 * 123, "output_tokens": 0}


def test_server_speaks_jev_wire_format():
    client, _ = fake_server({"B": 0.85, "A": 0.08, "C": 0.07, "Yes": 0.92, "No": 0.08})
    app = create_app(Jevify(OpenAICompatBackend(model="m", runtime="ollama", client=client)))
    tc = TestClient(app)
    req = json.load(open("examples/request.json"))
    r = tc.post("/v1/systemone", json=req, headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answers"]["department"]["type"] == "choice"
    assert body["answers"]["frustration"]["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert body["answers"]["is_urgent"]["noul"] == 0.92

    bad = tc.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": None}}}})
    assert bad.status_code == 400 and "2..26" in bad.json()["detail"]


def test_concurrency_fans_out_and_preserves_order():
    client, seen = fake_server({"A": 0.6, "B": 0.4})
    be = OpenAICompatBackend(model="m", runtime="vllm", client=client, concurrency=4)
    msgs = [[{"role": "user", "content": f"q{i}"}] for i in range(6)]
    out = be.next_token_batch(msgs)
    assert len(out) == 6 and all(nt.logprobs["A"] == math.log(0.6) for nt in out)
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}  # vllm no-think preset



def test_images_become_image_url_blocks_for_openai_servers():
    client, seen = fake_server({"Yes": 0.7, "No": 0.3})
    jev = Jevify(OpenAICompatBackend(model="m", runtime="vllm", client=client))
    out = jev.system_one({"image": "https://example.com/cat.jpg"}, {"dog": Noul("Is there a dog?")})
    assert out["answers"]["dog"]["noul"] == 0.7
    user = seen[0]["messages"][1]["content"]
    assert user[0] == {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
    assert user[1]["type"] == "text"
