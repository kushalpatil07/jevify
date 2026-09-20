# train/

The scripts that produced the published models. Commands are in the main README; this is what each
script does.

| script | what |
|---|---|
| `build_dataset.py` | pulls 16 public datasets, turns each row into (state, question, target distribution) in Jev's format, adds constructed long states up to 24k tokens. ~47k train rows, 800 eval rows. |
| `build_heldout.py` | 6 datasets not used in training (MASSIVE, TREC, PAWS, SMS spam, app reviews, subjectivity) plus 15 hand-written cases. For out-of-distribution numbers. |
| `train_lora.py` | LoRA on the attention projections. Loss = KL(target ‖ softmax over label tokens) at the answer position, plus a small term that keeps the next-token mass on the label tokens. Prompts come from `jevify.prompting.render`, so training and inference see the same text. Saves the best adapter by eval NLL and, with `--merge`, a merged model. |
| `eval.py` | accuracy, NLL, Brier, ECE per source for any jevify backend: a raw model, a merged model, `--adapter` on top of a base, or a running server. |
| `eval_image.py`, `probe_image.py` | the image checks: CIFAR-10 / Food-101 metrics, and Flickr photos with captions so you can read the answers. |
| `export.py` | merge the adapter, keep the image processor, push merged weights and the adapter to Hugging Face with a model card. |

## Where the targets come from

| kind | sources | why |
|---|---|---|
| hard labels, options subsetted and shuffled per row, instructions reworded | banking77, clinc, ag_news, dbpedia, emotion, mnli, boolq, irony, toxicchat, sst5, yelp, tweet_sentiment | cross-entropy on real labels is a proper scoring rule: the thing that minimizes it is the true P(label given state) |
| human label distributions | go_emotions (several raters per comment), measuring-hate-speech (3+ raters) | the only per-example "true probability" that exists |
| constructed states, 300 to 24k tokens | ticket threads with proportional targets, JSON records with a "not mentioned" option | long-context examples where the answer, and the ambiguity, is known because we wrote it |

No teacher model anywhere. Distilling a hosted model into yours is usually against its terms, and a
teacher's probabilities are just another model's opinion anyway.

## Data format

One JSON object per line. `question` is Jev's wire shape; `target` is keyed like jevify's answers
(option keys for choice, `"0".."N-1"` for score, `yes`/`no` for noul). `split` is `train` or `eval`.

```json
{"state": "...", "question": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}, "target": {"0": 0.1, "1": 0.7, "2": 0.2}, "split": "train", "source": "mine"}
```

## What the runs looked like

E4B: 1209 optimizer steps, 3h14m on one B200, 35 GB. 26B-A4B: 1215 steps, 4h52m, 110 GB. Both with
gradient checkpointing (the 26B does not fit without it at 32k-token micro-batches). Eval every 100
steps; NLL on the held-out rows flattened after about 600 steps on both.
