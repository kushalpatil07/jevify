"""Small OUT-OF-DISTRIBUTION eval set: sources that were not used in training (different domains,
label sets and formats), plus the hand-written spike cases. ~100 items per source.

usage: uv run --extra train python train/build_heldout.py --out data/heldout.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import onehot, pretty, row, subset_options, take  # noqa: E402

rng = random.Random(1)
N = 100


def massive_intent():
    ds = load_dataset("mteb/amazon_massive_intent", "en", split="test")
    keys = sorted(set(ds["label_text"]))
    out = []
    for i, ex in enumerate(take(ds, N, seed=1)):
        gold = ex["label_text"]
        opts = subset_options(keys, gold, k_max=10)
        q = {"type": "choice", "instructions": "Which voice-assistant intent does this utterance express?", "criteria": {pretty(o): None for o in opts}}
        out.append(row("ood_massive", "choice", ex["text"], q, onehot([pretty(o) for o in opts], pretty(gold)), "eval", i))
    return out


def trec():
    ds = load_dataset("SetFit/TREC-QC", split="test")
    desc = {"abbreviation": "an abbreviation or its expansion", "entities": "an entity (animal, color, product, ...)", "description and abstract concepts": "a description, definition or reason", "human beings": "a person or group", "locations": "a location", "numeric values": "a number, date, count or amount"}
    out = []
    for i, ex in enumerate(take(ds, N, seed=1)):
        gold = ex["label_coarse_text"]
        opts = list(desc)
        rng.shuffle(opts)
        q = {"type": "choice", "instructions": "What kind of answer is this question asking for?", "criteria": {k: desc[k] for k in opts}}
        out.append(row("ood_trec", "choice", ex["text"], q, onehot(list(q["criteria"]), gold), "eval", i))
    return out


def paws():
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    out = []
    for i, ex in enumerate(take(ds, N, seed=1)):
        q = {"type": "noul", "instructions": "Do these two sentences mean the same thing (paraphrases)?"}
        out.append(row("ood_paws", "noul", {"sentence_1": ex["sentence1"], "sentence_2": ex["sentence2"]}, q, {"yes": float(ex["label"]), "no": 1 - float(ex["label"])}, "eval", i))
    return out


def sms_spam():
    ds = load_dataset("ucirvine/sms_spam", split="train")
    out = []
    for i, ex in enumerate(take(ds, N, seed=1)):
        q = {"type": "noul", "instructions": "Is this SMS spam?", "criteria": {"true": "unsolicited promotion, scam or phishing", "false": "a normal personal or business message"}}
        out.append(row("ood_sms_spam", "noul", ex["sms"], q, {"yes": float(ex["label"]), "no": 1 - float(ex["label"])}, "eval", i))
    return out


def app_reviews():
    ds = load_dataset("sealuzh/app_reviews", split="train")
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    out = []
    for i, ex in enumerate(take(ds, N, seed=1)):
        q = {"type": "score", "instructions": "How many stars did this app reviewer give?", "criteria": levels}
        out.append(row("ood_app_reviews", "score", ex["review"], q, onehot([str(j) for j in range(5)], str(int(ex["star"]) - 1)), "eval", i))
    return out


def subj():
    ds = load_dataset("SetFit/subj", split="test")
    out = []
    for i, ex in enumerate(take(ds, N, seed=1)):
        q = {"type": "choice", "instructions": "Is this sentence subjective (an opinion) or objective (a fact)?", "criteria": {"objective": "states facts or plot", "subjective": "expresses opinion or judgement"}}
        gold = "objective" if ex["label_text"] == "objective" else "subjective"
        out.append(row("ood_subj", "choice", ex["text"], q, onehot(["objective", "subjective"], gold), "eval", i))
    return out


def spike_cases():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "spike"))
    from spike import CASES  # noqa: E402

    out = []
    for i, (kind, state, instr, opts, expected) in enumerate(CASES):
        if kind == "choice":
            q = {"type": "choice", "instructions": instr, "criteria": dict(opts)}
            keys = list(opts)
        elif kind == "score":
            q = {"type": "score", "instructions": instr, "criteria": list(opts)}
            keys = [str(j) for j in range(len(opts))]
        else:
            q = {"type": "noul", "instructions": instr}
            keys = ["yes", "no"]
        t = {k: (1.0 / len(expected) if k in expected else 0.0) for k in keys}
        out.append(row("spike", kind, state, q, t, "eval", i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/heldout.jsonl")
    args = ap.parse_args()
    rows = []
    for fn in (massive_intent, trec, paws, sms_spam, app_reviews, subj, spike_cases):
        got = fn()
        rows.extend(got)
        print(f"{fn.__name__:16} {len(got)}")
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
