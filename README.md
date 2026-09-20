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

| model | for | weights |
|---|---|---|
| `jevify-gemma4-e4b` | laptops (16 GB bf16; runs on a 24 GB Mac via MPS) | [`kushalpatil/jevify-gemma4-e4b`](https://huggingface.co/kushalpatil/jevify-gemma4-e4b) |
| `jevify-gemma4-26b-a4b` | GPUs / servers (26B MoE, 4B active, 52 GB bf16) | [`kushalpatil/jevify-gemma4-26b-a4b`](https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b) |

Merged weights, plus the LoRA adapters as `…-lora`. Both are still multimodal: an image can be the state.
Any other instruct model works too (it's just next-token logits) — it will just be over-confident.

## Use

```bash
pip install "jevify[transformers] @ git+https://github.com/kushalpatil07/jevify"   # or: uv sync --extra transformers
jevify serve --model kushalpatil/jevify-gemma4-e4b          # loads in-process; http://127.0.0.1:8000/v1/systemone
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
{"model": "kushalpatil/jevify-gemma4-e4b",
 "answers": {
  "is_urgent":   {"type": "noul", "noul": 0.7311},
  "department":  {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8354, "technical": 0.1645, "sales": 0.0001}, "confidence": 0.5919},
  "frustration": {"type": "score", "score": 0.5961, "legend": {"0": "calm", "1": "frustrated", "2": "very angry"},
                  "probabilities": {"0": 0.4064, "1": 0.5912, "2": 0.0024}, "confidence": 0.3708}},
 "usage": {"input_tokens": 256, "output_tokens": 0}}
```

(The raw model answers the same request with `billing: 0.997` and `noul: 1.0`.)

From Python:

```python
from jevify import Jevify, Noul, Choice, Score

jev = Jevify.from_transformers("kushalpatil/jevify-gemma4-e4b")
# or point at a server you already run:  Jevify.from_runtime("vllm", "kushalpatil/jevify-gemma4-26b-a4b")

res = jev.system_one(state, {
    "is_urgent":   Noul("Does this convey urgency?"),
    "department":  Choice("Which team should handle this?", {"billing": None, "technical": None, "sales": None}),
    "frustration": Score("How frustrated is the customer?", ["calm", "frustrated", "very angry"]),
})
res["answers"]["department"]["probabilities"]
```

With the official SDK: `TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=local`, then use it as normal
(`examples/use_typesafe_sdk.py`). One-off from the shell: `jevify ask --state "..." --choice "team: billing,technical,sales"`.

**Serving at scale:** run the HF model under vLLM or SGLang and use `--backend openai --runtime vllm`
(any OpenAI-compatible server that returns `logprobs` works: `vllm | sglang | llamacpp | lmstudio | ollama | mlx`).

## How it works

1. Each question becomes a tiny multiple-choice prompt after the shared `<state>` block
   (`A. billing  B. technical  C. sales — answer with the letter`; `Yes`/`No` for noul).
2. One forward pass; read the next-token distribution over the label tokens; renormalize.
   Score = `Σ i·pᵢ`. Confidence = `1 − H(p)/ln K`.
3. Many questions about one state cost one state prefill: the transformers backend keeps the
   state's KV cache and runs only each question's suffix (and, on models without sliding-window
   layers, packs all questions into a single forward pass with a tree attention mask — numerically
   identical, ~3–4× faster); server runtimes reuse their own prompt cache.

## Results

Held-out evaluation on **6 datasets never seen in training** (intent, question type, paraphrase,
spam, app-store stars, subjectivity). ECE = how far stated confidence is from actual accuracy;
NLL/Brier = proper scoring rules (lower is better).

| model | acc | mean conf | ECE | NLL | Brier |
|---|---|---|---|---|---|
| Gemma 4 E4B raw | 0.827 | 0.962 | 0.135 | 0.901 | 0.274 |
| **jevify-gemma4-e4b** | 0.844 | 0.849 | **0.043** | **0.435** | **0.237** |
| Gemma 4 26B-A4B raw | 0.847 | 0.986 | 0.142 | 1.846 | 0.285 |
| **jevify-gemma4-26b-a4b** | 0.834 | 0.875 | **0.061** | **0.450** | **0.241** |

Raw models are right ~84% of the time while claiming 96–99%. The trained ones claim what they
deliver. On the in-distribution held-out set (800 items, 16 sources) both go from ECE ≈ 0.22 to
≈ 0.03 and accuracy rises 7 points. The same holds with an image as the state (CIFAR-10 / Food-101,
never trained on): 26B confidence 0.999 → 0.93 at 98% accuracy.

## Train your own

`train/` has the whole pipeline — dataset from public labeled sets, LoRA training, eval, export.
See [train/README.md](train/README.md). Add your own `(state, question, target)` rows in the same
jsonl format to specialize it.

## Limitations

- Probabilities are honest *on average*; for a specific deployment, a few hundred of your own
  labeled rows through `train/` will beat the generic model.
- Server backends see the runtime's top-20 logprobs: options whose label falls outside get 0.
  Irrelevant below ~15 options; the transformers backend has no cap (≤26 options).
- Trained on text; images work through the Gemma vision tower (see `train/probe_image.py`) but
  weren't part of the calibration data. Thinking is disabled on reasoning models (it breaks the
  first-token trick).

MIT. The models are under the Gemma terms of use.
