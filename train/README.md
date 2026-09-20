# Training the jevify models

Everything here reproduces the two published models. No teacher model, no API calls — only
public datasets with real labels and constructed examples whose answers are known.

```bash
uv sync --extra train

# 1. dataset: ~47k (state, question, target-distribution) rows from 16 sources + 24k-token constructed states
uv run python train/build_dataset.py --out data/jevify_calib.jsonl

# 2. small out-of-distribution eval set (6 sources not used in training)
uv run python train/build_heldout.py --out data/heldout.jsonl

# 3. train (one GPU; the 26B-A4B needs ~110 GB in bf16 with gradient checkpointing)
CUDA_VISIBLE_DEVICES=0 uv run python train/train_lora.py --model google/gemma-4-26B-A4B-it \
    --data data/jevify_calib.jsonl --out runs/g26 --epochs 2 --max-len 32768 --merge

# 4. evaluate anything jevify can talk to (raw model, adapter, Ollama model...)
uv run python train/eval.py --data data/heldout.jsonl --backend transformers --model google/gemma-4-26B-A4B-it --adapter runs/g26/adapter_best

# 5. publish merged weights + adapter to Hugging Face
uv run python train/export.py --base google/gemma-4-26B-A4B-it --adapter runs/g26/adapter_best --repo kushalpatil/jevify-gemma4-26b-a4b
```

## What the model learns

The loss is `KL(target ‖ softmax(label_logits))` at the first answer token — exactly the quantity
jevify reads at inference — plus a small term keeping ~100% of next-token mass on the label tokens.
Prompts are `jevify.prompting.render()` output through the chat template, so train == inference.

Targets come from three kinds of sources:

| kind | sources | why |
|---|---|---|
| hard labels, randomized option subsets/order, paraphrased instructions | banking77, clinc, ag_news, dbpedia, emotion, mnli, boolq, irony, toxicchat, sst5, yelp, tweet_sentiment | cross-entropy on real labels is a proper scoring rule: the minimizer is the true P(label \| state) |
| human label *distributions* | go_emotions (multi-rater), measuring-hate-speech (3+ raters) | the only per-example "true probability" that exists |
| constructed states, 300–24k tokens | ticket threads with proportional targets, JSON records incl. "not mentioned" | long-context regime with answers (and ambiguity) known by construction |

`eval.py` reports accuracy, NLL, Brier and ECE. A calibrated model has mean confidence ≈ accuracy
and ECE near 0; raw Gemma 4 says 0.99 while being right 75% of the time.

## Data format

One JSON object per line; `question` is Jev's wire shape, `target` is keyed like jevify's answers
(option keys for choice, `"0".."N-1"` for score, `yes`/`no` for noul):

```json
{"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": null}}, "target": {"a": 0.8, "b": 0.2}, "split": "train", "source": "mine"}
```
