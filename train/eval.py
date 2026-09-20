"""Calibration eval for any jevify backend (Ollama / vLLM / transformers, base or LoRA-merged).

Per source and overall: accuracy (argmax vs argmax target), NLL, Brier, ECE (15 bins), JSD to target,
and the same after fitting a single temperature on half of the eval split (reported on the other half).

usage:
  uv run python train/eval.py --data data/jevify_calib.jsonl --runtime ollama --model gemma4:e4b --limit-per-source 100
  uv run --extra transformers python train/eval.py --data ... --backend transformers --model google/gemma-4-26B-A4B-it
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevify.cli import _build  # noqa: E402  reuse backend construction flags
from jevify.prompting import render  # noqa: E402
from jevify.questions import question_from_dict  # noqa: E402
from jevify.scoring import match_label_probs  # noqa: E402


def label_logits(backend, state, qdict):
    """Raw (unnormalized) log-probabilities of the label tokens, in target-key order."""
    q = question_from_dict(qdict)
    r = render(state, q)
    nt = backend.next_token(r.messages)
    raw, mass = match_label_probs(nt.logprobs, r.labels)
    eps = 1e-9
    lp = np.array([math.log(raw[lab] + eps) for lab in r.labels])
    return r.keys, lp, mass


class JevAPI:
    """Baseline: TypeSafe's hosted Jev, via its public API. Internal comparison only (their MCA bars publishing benchmarks)."""

    name = "jev"

    def __init__(self, model="jev-latest", concurrency=8):
        import os

        import httpx

        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("set TYPESAFE_API_KEY")
        self.model = model
        self.client = httpx.Client(base_url=os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai"), headers={"Authorization": f"Bearer {key}"}, timeout=120)
        self.concurrency = concurrency

    def probs(self, state, qdict):
        """-> (keys, probs) in the same key convention as the dataset targets."""
        body = {"state": state, "model": self.model, "questions": {"q": qdict}}
        for attempt in range(5):
            r = self.client.post("/v1/systemone", json=body)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            break
        r.raise_for_status()
        a = r.json()["answers"]["q"]
        if a["type"] == "noul":
            return ["yes", "no"], [a["noul"], 1 - a["noul"]]
        keys = list(qdict["criteria"]) if a["type"] == "choice" else [str(i) for i in range(len(qdict["criteria"]))]
        return keys, [a["probabilities"].get(k, 0.0) for k in keys]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def metrics(P, T):
    """P, T: lists of prob arrays (variable K). Returns dict of scalar metrics."""
    nll = brier = jsd = 0.0
    correct = 0
    conf, hit = [], []
    for p, t in zip(P, T):
        p = np.clip(p, 1e-9, 1)
        p = p / p.sum()
        nll += -float((t * np.log(p)).sum())
        brier += float(((p - t) ** 2).sum())
        m = 0.5 * (p + t)
        jsd += float(0.5 * (t * np.log(np.clip(t, 1e-9, 1) / m)).sum() + 0.5 * (p * np.log(p / m)).sum())
        ok = int(p.argmax() == t.argmax())
        correct += ok
        conf.append(float(p.max()))
        hit.append(ok)
    n = len(P)
    conf, hit = np.array(conf), np.array(hit)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return {"n": n, "acc": correct / n, "nll": nll / n, "brier": brier / n, "jsd": jsd / n, "ece": ece, "mean_conf": float(conf.mean())}


def fit_temperature(L, T, grid=np.exp(np.linspace(math.log(0.3), math.log(30), 120))):
    best = (float("inf"), 1.0)
    for temp in grid:
        nll = 0.0
        for lp, t in zip(L, T):
            p = softmax(lp / temp)
            nll += -float((t * np.log(np.clip(p, 1e-9, 1))).sum())
        if nll < best[0]:
            best = (nll, float(temp))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit-per-source", type=int, default=100)
    ap.add_argument("--exclude-sources", default="", help="comma-separated sources to skip (e.g. synthetic_long on a 4k-context runtime)")
    ap.add_argument("--out", default=None, help="write per-item logits jsonl here (for offline temperature fits)")
    # backend flags (same as `jevify ask`)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runtime", default="ollama")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--backend", default="openai", choices=["openai", "transformers", "jev"])
    ap.add_argument("--concurrency", type=int, default=8, help="parallel requests (jev backend)")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir to merge into the transformers model")
    ap.add_argument("--label", default=None, help="name for this run in the report")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--extra-body", default=None)
    args = ap.parse_args()

    if args.backend == "jev":
        be = JevAPI(model=args.model, concurrency=args.concurrency)
    else:
        be = _build(args).backend
        if args.adapter:
            from peft import PeftModel

            be.lm = PeftModel.from_pretrained(be.lm, args.adapter).merge_and_unload().eval()
            be.model = f"{args.model}+{Path(args.adapter).parent.name}/{Path(args.adapter).name}"

    per_src = collections.defaultdict(list)
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if r["source"] in args.exclude_sources.split(","):
                continue
            if r["split"] == args.split and len(per_src[r["source"]]) < args.limit_per_source:
                per_src[r["source"]].append(r)

    items = []  # (source, keys, logits, target, mass)
    t0 = time.time()
    n_total = sum(len(v) for v in per_src.values())
    done = 0
    out_f = open(args.out, "w") if args.out else None

    def score_one(r):
        if isinstance(be, JevAPI):
            keys, p = be.probs(r["state"], r["question"])
            return keys, np.log(np.array(p) + 1e-9), 1.0
        return label_logits(be, r["state"], r["question"])

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(args.concurrency if isinstance(be, JevAPI) else 1)
    for src, rows in per_src.items():
        for r, (keys, lp, mass) in zip(rows, pool.map(score_one, rows)):
            t = np.array([r["target"].get(k, 0.0) for k in keys])
            if t.sum() <= 0:
                continue
            t = t / t.sum()
            items.append((src, keys, lp, t, mass))
            if out_f:
                out_f.write(json.dumps({"id": r["id"], "source": src, "keys": keys, "logits": lp.tolist(), "target": t.tolist(), "mass": mass}) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{n_total}  {time.time()-t0:.0f}s", file=sys.stderr)
    if out_f:
        out_f.close()

    # temperature: fit on even-indexed items, report on odd (per-source temperature would overfit small n; one global T)
    fit = [x for i, x in enumerate(items) if i % 2 == 0]
    test = [x for i, x in enumerate(items) if i % 2 == 1]
    temp = fit_temperature([x[2] for x in fit], [x[3] for x in fit])

    def report(rows, label):
        P1 = [softmax(x[2]) for x in rows]
        PT = [softmax(x[2] / temp) for x in rows]
        T = [x[3] for x in rows]
        m1, mt = metrics(P1, T), metrics(PT, T)
        mass = float(np.mean([x[4] for x in rows]))
        print(f"{label:16} n={m1['n']:4d} mass={mass:.3f} | T=1: acc={m1['acc']:.3f} nll={m1['nll']:.3f} brier={m1['brier']:.3f} ece={m1['ece']:.3f} conf={m1['mean_conf']:.3f} | T={temp:.2f}: nll={mt['nll']:.3f} brier={mt['brier']:.3f} ece={mt['ece']:.3f} conf={mt['mean_conf']:.3f}")

    print(f"\n[{args.label or be.model}] model={be.model} backend={be.name} fitted temperature T={temp:.2f} (on {len(fit)} items; metrics below on the other {len(test)})\n")
    for src in per_src:
        rows = [x for x in test if x[0] == src]
        if rows:
            report(rows, src)
    report(test, "ALL")


if __name__ == "__main__":
    main()
