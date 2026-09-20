"""Ask three typed questions about one piece of state with a local model (Ollama by default).

    ollama pull gemma4:e4b
    uv run python examples/quickstart.py
"""
import json
import sys

from jevify import Choice, Jevify, Noul, Score

model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e4b"
jev = Jevify.from_runtime("ollama", model)  # calibration temperature picked up automatically

state = {
    "ticket_id": 8123,
    "customer_plan": "pro",
    "messages": [
        "Hi, I was charged twice this month and the second invoice has the wrong amount.",
        "Also the export button has been broken since Tuesday. This is getting frustrating.",
    ],
}

result = jev.system_one(
    state,
    {
        "team": Choice(
            "Which team should own this ticket?",
            {"billing": "payments, invoices, refunds", "technical": "bugs, outages, integrations", "sales": "pricing, upgrades, quotes"},
        ),
        "refund_requested": Noul("Does the customer explicitly ask for a refund?"),
        "frustration": Score("How frustrated is the customer?", ["calm", "mildly annoyed", "frustrated", "furious"]),
    },
    debug=True,
)
print(json.dumps(result, indent=2))
