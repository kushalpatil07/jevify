<div align="center">

<h1>Jevify</h1>

<h3>Make any open-weight model into a Jev API.</h3>

<p>
Keeps everything the base model has: its full context window (128k, 256k, 1M, whatever it ships with), image input, any architecture on Hugging Face.<br>
Two trained models included, plus the full recipe and scripts to train your own.
</p>

<p>
<a href="https://huggingface.co/kushalpatil/jevify-gemma4-e4b"><img src="https://img.shields.io/badge/🤗%20model-jevify--gemma4--e4b-yellow" alt="E4B"></a>
<a href="https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b"><img src="https://img.shields.io/badge/🤗%20model-jevify--gemma4--26b--a4b-yellow" alt="26B"></a>
<a href="https://github.com/kushalpatil07/jevify/actions"><img src="https://github.com/kushalpatil07/jevify/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python">
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
</p>

<p>
<a href="#quick-start">Quick start</a> ·
<a href="#how-it-works">How it works</a> ·
<a href="#why-the-models-are-trained">Results</a> ·
<a href="#models">Models</a> ·
<a href="#train-your-own">Train your own</a>
</p>

</div>

[Jev](https://docs.typesafe.ai/api) is TypeSafe's API: send state and typed questions, get
probabilities back. Jevify serves the same request and response format from a model you run, so
the `typesafe-sdk` works unchanged. Any causal LM from Hugging Face works; the two Gemma 4 models
below are trained to give calibrated probabilities.

## Quick start

```bash
pip install "jevify[transformers] @ git+https://github.com/kushalpatil07/jevify"
jevify serve --model kushalpatil/jevify-gemma4-e4b      # POST http://127.0.0.1:8000/v1/systemone
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
  "is_urgent":   {"type": "noul",   "noul": 0.7311},
  "department":  {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8354, "technical": 0.1645, "sales": 0.0001}, "confidence": 0.5919},
  "frustration": {"type": "score",  "score": 0.5961, "probabilities": {"0": 0.4064, "1": 0.5912, "2": 0.0024}, "confidence": 0.3708}},
 "usage": {"input_tokens": 256, "output_tokens": 0}}
```

```python
from jevify import Jevify, Noul, Choice, Score

jev = Jevify.from_transformers("kushalpatil/jevify-gemma4-e4b")
res = jev.system_one(state, {
    "is_urgent":   Noul("Does this convey urgency?"),
    "department":  Choice("Which team should handle this?", {"billing": None, "technical": None, "sales": None}),
    "frustration": Score("How frustrated is the customer?", ["calm", "frustrated", "very angry"]),
})
```

| type | question | answer |
|---|---|---|
| `noul` | is this true? | `P(yes)` |
| `choice` | which of these? | distribution over your options, confidence |
| `score` | how much, on ordered levels? | distribution over levels, expected level |

State can be a string, a JSON object, or an image (`{"image": "photo.jpg"}`, a URL, a data URL).
For the official SDK, set `TYPESAFE_BASE_URL=http://localhost:8000` and `TYPESAFE_API_KEY=local`.
Image demo: [examples/hotdog](examples/hotdog), the Not Hotdog app, with a live camera mode.

## How it works

Each question becomes a short multiple-choice prompt after the state:

```
<state>
Help! My payouts have been failing for 3 days.
</state>

Which team should handle this?

Options:
A. billing — payments, refunds
B. technical — bugs, outages
C. sales — pricing, upgrades

Answer with the letter of the single best option.
```

One forward pass. Read the next-token distribution at the last position, keep the entries for
`A`, `B`, `C`, renormalize. `noul` uses `Yes`/`No`. `score` uses a letter per level and reports the
expected level. Confidence is 1 minus normalized entropy. `output_tokens` is always 0.

Several questions about one state cost one state prefill. The state's KV cache is computed once,
copied per question along the batch dimension, and the question suffixes run as one batched forward:

```
STATE ──► KV cache (once)
            row 1: [cache] + Q1 ─┐
            row 2: [cache] + Q2 ─┼─► one forward ─► N answers
            row 3: [cache] + Q3 ─┘
```

Each row is "state + its question", so the result equals asking one at a time (identical in float32).
Full-attention models can instead pack the questions into one row with a tree-shaped mask, which
shares a single cache copy for very long states. Server backends get the same effect from their
prompt cache.

B200, eager transformers, 1,700-token state cached:

| | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| jevify-gemma4-e4b | 40 ms | 48 ms | 76 ms |
| jevify-gemma4-26b-a4b | 61 ms | 72 ms | 112 ms |

First prefill of that state adds 10 ms (E4B) or 25 ms (26B). The 40 ms floor is Python overhead;
vLLM or `torch.compile` removes most of it. A 24 GB M-series Mac prefills the same state in 4.5 s and
answers a question in 0.3 s.

Context and images come from the base model. Gemma 4 E4B takes 128k tokens, the 26B 256k, and both
accept an image as the state.

## Why the models are trained

A chat model's output layer predicts the next token. It does not estimate how often it is right,
and RLHF makes it say 0.99 for most answers. On 800 held-out items from 16 datasets:

| | accuracy | stated confidence |
|---|---|---|
| Gemma 4 E4B, raw | 0.745 | 0.963 |
| Gemma 4 26B-A4B, raw | 0.757 | 0.991 |
| Jev 1.13 (API) | 0.756 | 0.872 |

We trained the output distribution directly: a LoRA on the attention layers, loss = KL between a
target distribution and the model's distribution over the label tokens, at the position Jevify
reads. No teacher model. Targets came from public classification sets with real labels (options
subsetted, shuffled and reworded per row), from sets with several raters per item (go_emotions,
measuring-hate-speech), and from constructed documents up to 24k tokens with known answers,
including 50/50 and 75/25 cases.

One run per model, 47k rows, 2 epochs, one GPU. ECE is the gap between stated confidence and
accuracy. NLL and Brier are proper scoring rules, lower is better. Jev was run through its API on
the same items; its probabilities are rounded to two decimals, so its NLL uses a 0.005 floor.

Held-out rows from the training sources (800 items):

| | accuracy | confidence | ECE | NLL | Brier |
|---|---|---|---|---|---|
| E4B raw | 0.745 | 0.963 | 0.218 | 1.83 | 0.412 |
| E4B jevified | 0.823 | 0.823 | 0.028 | 0.438 | 0.193 |
| 26B raw | 0.757 | 0.991 | 0.235 | 3.18 | 0.425 |
| 26B jevified | 0.821 | 0.829 | 0.032 | 0.422 | 0.188 |
| Jev 1.13 | 0.756 | 0.872 | 0.118 | 0.737 | 0.302 |

Six datasets not used in training (intents, question types, paraphrase, spam, app-store stars,
subjectivity; 307 items):

| | accuracy | confidence | ECE | NLL | Brier |
|---|---|---|---|---|---|
| E4B raw | 0.827 | 0.962 | 0.135 | 0.901 | 0.274 |
| E4B jevified | 0.844 | 0.849 | 0.043 | 0.435 | 0.237 |
| 26B raw | 0.847 | 0.986 | 0.142 | 1.846 | 0.285 |
| 26B jevified | 0.834 | 0.875 | 0.061 | 0.450 | 0.241 |
| Jev 1.13 | 0.866 | 0.879 | 0.037 | 0.387 | 0.197 |

On unseen datasets Jev is
ahead by 2 to 3 accuracy points and on NLL; the gap is concentrated in paraphrase detection (PAWS),
where our models score 0.54 to 0.58 against Jev's 0.86. On the other five datasets they are even.

## Models

| model | params | memory (bf16) | runs on | adapter |
|---|---|---|---|---|
| [jevify-gemma4-e4b](https://huggingface.co/kushalpatil/jevify-gemma4-e4b) | 8B | 16 GB | one GPU, or a 24 GB Mac | [lora](https://huggingface.co/kushalpatil/jevify-gemma4-e4b-lora) |
| [jevify-gemma4-26b-a4b](https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b) | 26B MoE, 4B active | 52 GB | one 80 GB GPU | [lora](https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b-lora) |

Merged weights load with transformers or vLLM. Adapters are 140 to 180 MB.

## Train your own

```bash
uv sync --extra train
uv run python train/build_dataset.py --out data/jevify_calib.jsonl        # 47k rows, 16 sources
uv run python train/build_heldout.py --out data/heldout.jsonl             # 6 unseen sources
CUDA_VISIBLE_DEVICES=0 uv run python train/train_lora.py --model google/gemma-4-E4B-it \
    --data data/jevify_calib.jsonl --out runs/e4b --epochs 2 --max-len 32768 --merge
uv run python train/eval.py --data data/heldout.jsonl --backend transformers --model runs/e4b/merged
uv run python train/export.py --base google/gemma-4-E4B-it --adapter runs/e4b/adapter_best --repo you/your-model
```

E4B trains in 3 hours on one B200, the 26B in 5. Any causal LM on Hugging Face works as the base.
Add your own rows to the same file, one JSON object per line:

```json
{"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": null}}, "target": {"a": 0.8, "b": 0.2}, "split": "train", "source": "mine"}
```

Training setup is pretty naive. By smart training you should be easily be able to beat Jev. 

Details of each script and data source: [train/README.md](train/README.md).

## Serving

Default is in-process transformers. For throughput, run the model under vLLM or SGLang:

```bash
jevify serve --backend openai --runtime vllm --model kushalpatil/jevify-gemma4-26b-a4b --concurrency 8
```

Any OpenAI-compatible server that returns `logprobs` works (`vllm`, `sglang`, `llamacpp`,
`lmstudio`, `ollama`, `mlx`). Servers expose the top 20 logprobs, so an option outside the top 20
gets probability 0; this does not happen below about 15 options.

## Limits

- Probabilities are calibrated on average. For one task, a few hundred of your own rows do better.
- Thinking mode is off. The method reads the first token, and a thinking model's first token is `<think>`.
- Up to 26 options per `choice`, 2 to 10 levels per `score`.
- Evaluated on English text and two image benchmarks.

MIT. Models are under the Gemma terms of use.
