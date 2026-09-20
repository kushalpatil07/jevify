# Jevify

**Turn any open-weight model into a Jev API.**

One forward pass in, calibrated probabilities out. Works with any causal LM on Hugging Face and
keeps whatever the base model can do: its full context window, images if it has a vision tower.

Jev is TypeSafe's API. You give it some state and a few typed questions, and it returns
probabilities instead of text. Jevify does the same thing with a model you run yourself. The
request and response format is the same, so the `typesafe-sdk` works against it without changes.

It comes with two Gemma 4 models we trained to give good probabilities. Any other instruct model
from Hugging Face works too, it will just be over-confident (more on that below).

## Try it

```bash
pip install "jevify[transformers] @ git+https://github.com/kushalpatil07/jevify"
jevify serve --model kushalpatil/jevify-gemma4-e4b      # http://127.0.0.1:8000/v1/systemone
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

From Python:

```python
from jevify import Jevify, Noul, Choice, Score

jev = Jevify.from_transformers("kushalpatil/jevify-gemma4-e4b")
res = jev.system_one(state, {
    "is_urgent":   Noul("Does this convey urgency?"),
    "department":  Choice("Which team should handle this?", {"billing": None, "technical": None, "sales": None}),
    "frustration": Score("How frustrated is the customer?", ["calm", "frustrated", "very angry"]),
})
res["answers"]["department"]["probabilities"]
```

The state can be a string, a JSON object, or an image (`{"image": "photo.jpg", "note": "..."}`,
a URL, or a data URL). To use the official SDK, set `TYPESAFE_BASE_URL=http://localhost:8000` and
`TYPESAFE_API_KEY=local`. `jevify ask --state "..." --choice "team: billing,technical,sales"` works
from the shell.

Three question types. You can mix them in one call.

| type | you ask | you get |
|---|---|---|
| `noul` | is this true? | `P(yes)` |
| `choice` | which of these? | a distribution over your options, plus a confidence |
| `score` | how much, on these ordered levels? | a distribution over the levels, plus the expected level |

## How it works

There is no text generation anywhere. Here is the whole trick.

Each question is written as a small multiple-choice prompt after the state:

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

The model runs one forward pass. At the last position it has a distribution over its whole
vocabulary for the next token. We take the probabilities of `A`, `B` and `C` out of it, renormalize
them, and that is the answer. For `noul` the labels are `Yes` and `No`. For `score` the labels are
letters for each level, and the score is the expected level index. Confidence is one minus the
normalized entropy of the distribution. `output_tokens` is always 0.

When you ask several questions about one state, the state is only processed once. The
transformers backend keeps the state's KV cache and runs each question's few dozen tokens on top
of it. On models without sliding-window attention it goes one step further and packs all the
questions into a single forward pass with a tree-shaped attention mask, so every question sees the
state and itself but not the other questions:

```
                STATE      Q1     Q2     Q3
      STATE     causal
      Q1        ■■■■■■   causal
      Q2        ■■■■■■           causal
      Q3        ■■■■■■                  causal
```

This gives the same numbers as asking the questions one at a time (we test that), it is just one
kernel launch instead of N. Server backends (vLLM, llama.cpp, ...) get the same effect from their
own prompt cache.

Speed, measured on one B200 with plain Hugging Face transformers, no compilation, 1,700-token state:

| | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| jevify-gemma4-e4b | 51 ms | 134 ms | 423 ms |
| jevify-gemma4-26b-a4b | 88 ms | 219 ms | 658 ms |

About 40 ms per question on the E4B and 60 ms on the 26B after the state is cached. That is the
Python overhead floor of eager transformers, not the GPU; vLLM or `torch.compile` would cut it
further. On a 24 GB M-series Mac (MPS) the E4B takes about 4.5 s to prefill the same state and then
0.3 s per question.

Because the model is just doing a normal forward pass, everything the base model supports still
works. Gemma 4 E4B takes 128k tokens of state, the 26B takes 256k. Both have a vision tower, so an
image can be the state and the questions are asked about the picture.

## Why raw models give bad probabilities

A chat model's last layer is trained to pick the next token, not to say how sure it is. After
instruction tuning and RLHF the distribution collapses. Ask a raw Gemma 4 which team should handle
the payouts message above and it says `billing: 0.997`. Ask it something it has no way of knowing
and it still says 0.99 for whatever it picks. The numbers look like probabilities but they are
not.

You can measure this. On 800 held-out items from 16 datasets:

| | accuracy | average stated confidence |
|---|---|---|
| Gemma 4 E4B, raw | 0.745 | 0.963 |
| Gemma 4 26B-A4B, raw | 0.757 | 0.991 |

Right three times out of four, sure 99 times out of 100.

So we trained the output distribution directly. A LoRA on the attention layers, and the loss is
the KL divergence between a target distribution and the model's distribution over the label
tokens, at exactly the position Jevify reads. No teacher model. The targets come from three places:

- public classification datasets with real labels (banking77, clinc, ag_news, boolq, mnli, sst5, yelp, toxicchat, ...),
  with the options subsetted, shuffled and reworded per example so the model can't memorize label sets or positions
- datasets where several people labeled each item (go_emotions, measuring-hate-speech), which give a real
  human distribution per item
- long documents we construct ourselves, up to 24k tokens, where we know the answer because we
  wrote it, including deliberately ambiguous ones with 50/50 or 75/25 targets

One run each, about 47k examples, two epochs, one GPU. Here is what changed. ECE is the average gap
between stated confidence and actual accuracy. NLL and Brier are proper scoring rules; lower is
better and confident mistakes cost the most.

Held-out rows from the training datasets (800 items):

| | accuracy | confidence | ECE | NLL | Brier |
|---|---|---|---|---|---|
| E4B raw | 0.745 | 0.963 | 0.218 | 1.83 | 0.412 |
| E4B jevified | 0.823 | 0.823 | 0.028 | 0.438 | 0.193 |
| 26B raw | 0.757 | 0.991 | 0.235 | 3.18 | 0.425 |
| 26B jevified | 0.821 | 0.829 | 0.032 | 0.422 | 0.188 |

Six datasets the models never saw during training (intents, question types, paraphrase, spam,
app-store stars, subjectivity; 307 items):

| | accuracy | confidence | ECE | NLL | Brier |
|---|---|---|---|---|---|
| E4B raw | 0.827 | 0.962 | 0.135 | 0.901 | 0.274 |
| E4B jevified | 0.844 | 0.849 | 0.043 | 0.435 | 0.237 |
| 26B raw | 0.847 | 0.986 | 0.142 | 1.846 | 0.285 |
| 26B jevified | 0.834 | 0.875 | 0.061 | 0.450 | 0.241 |

Stated confidence now matches accuracy, and accuracy went up too. The training was text only, but
it carried over to images: on CIFAR-10 and Food-101 the 26B goes from 0.999 confidence to 0.93 at
98% accuracy.

## Models

| model | size | runs on | weights | adapter |
|---|---|---|---|---|
| jevify-gemma4-e4b | 8B (bf16 16 GB) | one GPU, or a 24 GB Mac via MPS | [kushalpatil/jevify-gemma4-e4b](https://huggingface.co/kushalpatil/jevify-gemma4-e4b) | [-lora](https://huggingface.co/kushalpatil/jevify-gemma4-e4b-lora) |
| jevify-gemma4-26b-a4b | 26B MoE, 4B active (bf16 52 GB) | one 80 GB GPU | [kushalpatil/jevify-gemma4-26b-a4b](https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b) | [-lora](https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b-lora) |

The merged weights load with transformers or vLLM like the base models. The adapters are 140 to
180 MB if you already have the base weights. Both stay multimodal.

## Train your own

Everything we used is in `train/`. Public data only, no API calls.

```bash
uv sync --extra train
uv run python train/build_dataset.py --out data/jevify_calib.jsonl        # ~47k rows from 16 sources
uv run python train/build_heldout.py --out data/heldout.jsonl             # 6 unseen sources for eval
CUDA_VISIBLE_DEVICES=0 uv run python train/train_lora.py --model google/gemma-4-E4B-it \
    --data data/jevify_calib.jsonl --out runs/e4b --epochs 2 --max-len 32768 --merge
uv run python train/eval.py --data data/heldout.jsonl --backend transformers --model runs/e4b/merged
uv run python train/export.py --base google/gemma-4-E4B-it --adapter runs/e4b/adapter_best --repo you/your-model
```

The E4B trains in about 3 hours on one B200, the 26B in about 5. Any causal LM on Hugging Face
works as the base.

Your own rows go in the same file. One JSON object per line; the question is in Jev's format and
the target is keyed like Jevify's answers:

```json
{"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": null}}, "target": {"a": 0.8, "b": 0.2}, "split": "train", "source": "mine"}
```

We did one straightforward run and the numbers moved a lot. There is room, and most of it is in the
data and the recipe, not in the code. Things we would try next, in order:

1. Add paraphrase and NLI style pairs. Our models lose the most on PAWS, where two sentences differ
   by a word or two. That is the one task where the label depends on reading carefully rather than
   recognizing a category.
2. Train for one epoch, not two. Checkpoints from early in the run scored slightly better on the
   unseen datasets than the final ones.
3. Lower learning rate for the 26B. Its gradient norms were spiky at 1e-4.
4. A few hundred rows of your own questions. Generic calibration gets you to "0.8 means about 80%
   on average". Rows from your own distribution get you there for your questions specifically.
5. Pack several questions per state during training with the same tree mask used at inference.
   The long-state examples would cost a fraction of what they cost now.
6. A bigger base model.

## Serving

The default is in-process transformers. For throughput, run the Hugging Face model under vLLM or
SGLang and point Jevify at it:

```bash
jevify serve --backend openai --runtime vllm --model kushalpatil/jevify-gemma4-26b-a4b --concurrency 8
```

Any OpenAI-compatible server that returns `logprobs` works this way (`vllm`, `sglang`, `llamacpp`,
`lmstudio`, `ollama`, `mlx`). Server backends only see the top 20 logprobs per position, so an option
whose label falls outside the top 20 gets probability 0. That does not happen below about 15
options. The transformers backend has no such cap.

## Limits

- The probabilities are good on average. For one specific task, a few hundred of your own labeled
  rows will do better than the generic model.
- Reasoning/thinking mode is turned off. The trick reads the first token, and a thinking model's
  first token is `<think>`.
- Up to 26 options per `choice` (letters), 2 to 10 levels per `score`.
- We evaluated on English text and a couple of image benchmarks. Other languages should work but
  we did not measure them.

MIT license. The models are released under the Gemma terms of use.
