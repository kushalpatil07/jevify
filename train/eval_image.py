"""Multimodal probe: image as the state, jevify-style questions, label-token probabilities.

Compares a base model vs base+LoRA on small image tasks the LoRA never saw (it was trained on
text only). Reports acc / NLL / Brier / ECE / mean confidence, and per-item examples.

  CUDA_VISIBLE_DEVICES=5 uv run --extra train python train/eval_image.py --model google/gemma-4-E4B-it [--adapter runs/e4b_snap] --n 100
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevify.prompting import SYSTEM_PROMPT, render  # noqa: E402
from jevify.questions import Choice, Noul  # noqa: E402
from jevify.scoring import match_label_probs, normalize  # noqa: E402

CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ANIMALS = {"bird", "cat", "deer", "dog", "frog", "horse"}


def build_items(n, seed=0):
    from datasets import load_dataset

    rng = random.Random(seed)
    items = []
    ds = load_dataset("uoft-cs/cifar10", split="test").shuffle(seed=seed).select(range(n))
    for ex in ds:
        gold = CIFAR[ex["label"]]
        opts = CIFAR[:]
        rng.shuffle(opts)
        items.append(("cifar10_choice", ex["img"], Choice("What is the main object in the image?", {o: None for o in opts}), {o: float(o == gold) for o in opts}))
        yes = gold in ANIMALS
        items.append(("cifar10_animal", ex["img"], Noul("Does the image show an animal?"), {"yes": float(yes), "no": float(not yes)}))
    try:
        food = load_dataset("ethz/food101", split="validation").shuffle(seed=seed).select(range(n))
        names = food.features["label"].names
        for ex in food:
            gold = names[ex["label"]].replace("_", " ")
            others = [x.replace("_", " ") for x in names if x.replace("_", " ") != gold]
            rng.shuffle(others)
            opts = others[:7] + [gold]
            rng.shuffle(opts)
            items.append(("food101_choice", ex["image"], Choice("Which dish is shown?", {o: None for o in opts}), {o: float(o == gold) for o in opts}))
    except Exception as e:  # dataset unavailable -> skip
        print("food101 skipped:", str(e)[:100], file=sys.stderr)
    return items


def metrics(P, T):
    nll = brier = 0.0
    conf, hit = [], []
    for p, t in zip(P, T):
        p = np.clip(p, 1e-9, 1)
        p = p / p.sum()
        nll += -float((t * np.log(p)).sum())
        brier += float(((p - t) ** 2).sum())
        conf.append(float(p.max()))
        hit.append(int(p.argmax() == t.argmax()))
    conf, hit = np.array(conf), np.array(hit)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    n = len(P)
    return dict(n=n, acc=hit.mean(), conf=conf.mean(), ece=ece, nll=nll / n, brier=brier / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    label = f"{args.model}" + (f" + {args.adapter}" if args.adapter else " (raw)")

    items = build_items(args.n)
    by_task: dict[str, tuple[list, list]] = {}
    shown = 0
    for task, img, q, target in items:
        r = render("[the attached image]", q, SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": r.messages[0]["content"]}]},
            {"role": "user", "content": [{"type": "image", "image": img.convert("RGB")}, {"type": "text", "text": r.messages[1]["content"]}]},
        ]
        inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        top = torch.topk(lp, 64)
        table = {proc.tokenizer.decode([int(i)]): float(v) for v, i in zip(top.values, top.indices)}
        raw, mass = match_label_probs(table, r.labels)
        probs = normalize(raw)
        p = np.array([probs[lab] for lab in r.labels])
        t = np.array([target[k] for k in r.keys])
        by_task.setdefault(task, ([], []))[0].append(p)
        by_task[task][1].append(t)
        if shown < args.show and task != "cifar10_animal":
            best = r.keys[int(p.argmax())]
            truth = r.keys[int(t.argmax())]
            top3 = sorted(zip(r.keys, p), key=lambda kv: -kv[1])[:3]
            print(f"  [{task}] truth={truth:<14} pred={best:<14} {'✓' if best == truth else '✗'} mass={mass:.2f} top3={[(k, round(float(v), 2)) for k, v in top3]}")
            shown += 1

    print(f"\n== {label}")
    allP, allT = [], []
    for task, (P, T) in by_task.items():
        m = metrics(P, T)
        allP += P
        allT += T
        print(f"{task:16} n={m['n']:3d} acc={m['acc']:.3f} conf={m['conf']:.3f} ece={m['ece']:.3f} nll={m['nll']:.3f} brier={m['brier']:.3f}")
    m = metrics(allP, allT)
    print(f"{'ALL':16} n={m['n']:3d} acc={m['acc']:.3f} conf={m['conf']:.3f} ece={m['ece']:.3f} nll={m['nll']:.3f} brier={m['brier']:.3f}")


if __name__ == "__main__":
    main()
