"""Use TypeSafe's official Python SDK against a local jevify server.

    uv run jevify serve --runtime ollama --model gemma4:e4b
    TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=local uv run python examples/use_typesafe_sdk.py
"""
import os

os.environ.setdefault("TYPESAFE_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TYPESAFE_API_KEY", "local")

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient  # noqa: E402

client = TypeSafeClient()
result = client.system_one(
    "Help! My payouts have been failing for 3 days.",
    {
        "is_urgent": Noul(instructions="Does this convey urgency?"),
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"},
        ),
        "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Frustrated", "Very angry"]),
    },
)
print("model:      ", result.model)
print("is_urgent:  ", result.nouls["is_urgent"].noul)
print("department: ", result.choices["department"].choice, result.choices["department"].probabilities, "conf", result.choices["department"].confidence)
print("frustration:", result.scores["frustration"].score, result.scores["frustration"].probabilities)
