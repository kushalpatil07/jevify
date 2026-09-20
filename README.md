# jevify

Run a local LLM as a **probabilistic decision API**: state in, typed probability distributions out.
One prefill, **zero decode**. Same wire format as [Jev](https://docs.typesafe.ai/api), so the
`typesafe-sdk` works against it unchanged.

```
state + questions ──► ONE PREFILL ──► next-token logits of the answer labels ──► softmax ──► probabilities
```

Three question types, mixable in one call:

| type | asks | returns |
|---|---|---|
| `noul` | is this true? | `P(yes)` |
| `choice` | which option? | distribution over your options + confidence |
| `score` | how much, on these ordered levels? | distribution over levels + expected value |

## Models

Two Gemma 4 models fine-tuned (LoRA) so the probabilities are **honest** — a stated 0.8 is right
about 80% of the time. Raw chat models say 0.99 and are right 75% of the time.

| model | for | HF | Ollama |
|---|---|---|---|
| `jevify-gemma4-e4b` | laptops (8 GB) | `kushalpatil/jevify-gemma4-e4b` | `ollama pull kushalpatil/jevify-gemma4-e4b` |
| `jevify-gemma4-26b-a4b` | workstations / servers (26B MoE, 4B active) | `kushalpatil/jevify-gemma4-26b-a4b` | `ollama pull kushalpatil/jevify-gemma4-26b-a4b` |

Any other instruct model works too (it's just next-token logits), it will just be over-confident.

## Use

```bash
ollama pull kushalpatil/jevify-gemma4-e4b
uv sync                                                        # or: pip install -e .
uv run jevify serve --runtime ollama --model kushalpatil/jevify-gemma4-e4b   # http://127.0.0.1:8000/v1/systemone
```

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "payments, refunds", "technical": "bugs, outages", "sales": "pricing, upgrades"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very angry"]}
  }}'
```

```json
{"answers": {
  "is_urgent":   {"type": "noul", "noul": 0.93},
  "department":  {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.79, "technical": 0.2, "sales": 0.01}, "confidence": 0.54},
  "frustration": {"type": "score", "score": 1.08, "legend": {"0": "calm", "1": "frustrated", "2": "very angry"},
                  "probabilities": {"0": 0.03, "1": 0.86, "2": 0.11}, "confidence": 0.58}},
 "usage": {"input_tokens": 280, "output_tokens": 0}}
```

From Python:

```python
from jevify import Jevify, Noul, Choice, Score

jev = Jevify.from_runtime("ollama", "kushalpatil/jevify-gemma4-e4b")
# or in-process, no server:  Jevify.from_transformers("kushalpatil/jevify-gemma4-e4b")

res = jev.system_one(state, {
    "is_urgent":   Noul("Does this convey urgency?"),
    "department":  Choice("Which team should handle this?", {"billing": None, "technical": None, "sales": None}),
    "frustration": Score("How frustrated is the customer?", ["calm", "frustrated", "very angry"]),
})
res["answers"]["department"]["probabilities"]
```

With the official SDK: `TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=local`, then use it as normal
(`examples/use_typesafe_sdk.py`). One-off from the shell: `jevify ask --state "..." --choice "team: billing,technical,sales"`.

**Runtimes:** `--runtime ollama | llamacpp | lmstudio | vllm | sglang | mlx` (any OpenAI-compatible server
that returns `logprobs`), or `--backend transformers` to load the model in-process. For states longer
than 4k tokens on Ollama, start it with `OLLAMA_CONTEXT_LENGTH=32768`.

## How it works

1. Each question becomes a tiny multiple-choice prompt after the shared `<state>` block
   (`A. billing  B. technical  C. sales — answer with the letter`; `Yes`/`No` for noul).
2. One forward pass; read the next-token distribution over the label tokens; renormalize.
   Score = `Σ i·pᵢ`. Confidence = `1 − H(p)/ln K`.
3. Many questions about one state cost one state prefill: server runtimes reuse their prompt
   cache (Ollama reports `cached_tokens`); the transformers backend packs all questions into a
   single forward pass with a tree attention mask (each question sees state + itself only).
   Numerically identical to asking one at a time, ~3–4× faster.

## Results

Held-out evaluation on **6 datasets never seen in training** (intent, question type, paraphrase,
spam, app-store stars, subjectivity). ECE = how far stated confidence is from actual accuracy;
NLL/Brier = proper scoring rules (lower is better).

| model | acc | mean conf | ECE | NLL | Brier |
|---|---|---|---|---|---|
| Gemma 4 E4B raw | 0.827 | 0.962 | 0.135 | 0.901 | 0.274 |
| **jevify-gemma4-e4b** | 0.857 | 0.859 | **0.037** | **0.401** | **0.213** |
| Gemma 4 26B-A4B raw | 0.847 | 0.986 | 0.142 | 1.846 | 0.285 |
| **jevify-gemma4-26b-a4b** | _final numbers pending_ | | | | |

Raw models are right ~84% of the time while claiming ~97–99%. The trained ones claim ~86% and are
right ~86%.

## Train your own

`train/` has the whole pipeline — dataset from public labeled sets, LoRA training, eval, export.
See [train/README.md](train/README.md). Add your own `(state, question, target)` rows in the same
jsonl format to specialize it.

## Limitations

- Probabilities are honest *on average*; for a specific deployment, a few hundred of your own
  labeled rows through `train/` will beat the generic model.
- Server backends see the runtime's top-20 logprobs: options whose label falls outside get 0.
  Irrelevant below ~15 options; the transformers backend has no cap (≤26 options).
- Text-only state. Thinking is disabled on reasoning models (it breaks the first-token trick).

MIT. The models are under the Gemma terms of use.
