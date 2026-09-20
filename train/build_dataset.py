"""Build the calibration-training dataset: (state, question, target distribution) triples in Jev's shape.

Sources of *probability* signal (no teacher model anywhere):
  1. hard-labeled classification sets, with per-item option subsetting + shuffling + instruction
     paraphrase (cross-entropy on real labels is a proper scoring rule -> calibrated in expectation)
  2. multi-annotator sets -> real human label distributions per item
  3. constructed long states whose answers (and ambiguity) are known by construction

Output: jsonl rows {id, source, kind, split, state, question, target}
  question = {"type": "noul"|"choice"|"score", "instructions": str, "criteria": ...}   (Jev wire shape)
  target   = {label_key: prob}   keys = option keys / "0".."N-1" / "yes","no"

usage: uv run --extra train python train/build_dataset.py --out data/jevify_calib.jsonl [--per-source 3000] [--eval-per-source 300]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

rng = random.Random(0)

# ----------------------------------------------------------------------------- helpers
def paraphrase(options: list[str]) -> str:
    return rng.choice(options)


def subset_options(all_keys: list[str], gold: str, k_max: int = 12) -> list[str]:
    """Random subset of 2..k_max options that always contains `gold`, in random order."""
    k = rng.randint(2, min(k_max, len(all_keys)))
    others = [o for o in all_keys if o != gold]
    rng.shuffle(others)
    opts = others[: k - 1] + [gold]
    rng.shuffle(opts)
    return opts


def onehot(keys: list[str], gold: str) -> dict[str, float]:
    return {k: (1.0 if k == gold else 0.0) for k in keys}


def pretty(label: str) -> str:
    return label.replace("_", " ").strip()


def row(source, kind, state, question, target, split, idx):
    return {
        "id": f"{source}-{idx}",
        "source": source,
        "kind": kind,
        "split": split,
        "state": state,
        "question": question,
        "target": target,
    }


def take(ds, n, seed=0):
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


# ----------------------------------------------------------------------------- sources
def src_banking77(n, n_eval):
    ds = load_dataset("mteb/banking77", split="train")
    keys = sorted(set(ds["label_text"]))
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        gold = ex["label_text"]
        opts = subset_options(keys, gold)
        q = {
            "type": "choice",
            "instructions": paraphrase(["What is the customer asking about?", "Which banking intent best matches this message?", "Classify the intent of this customer message."]),
            "criteria": {pretty(o): None for o in opts},
        }
        out.append(row("banking77", "choice", ex["text"], q, onehot([pretty(o) for o in opts], pretty(gold)), "eval" if i < n_eval else "train", i))
    return out


def src_clinc(n, n_eval):
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ds.features["intent"].names
    ds = ds.filter(lambda e: names[e["intent"]] != "oos")
    keys = [x for x in names if x != "oos"]
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        gold = names[ex["intent"]]
        opts = subset_options(keys, gold)
        q = {"type": "choice", "instructions": paraphrase(["What does the user want?", "Which intent does this utterance express?", "Pick the intent that best fits the request."]), "criteria": {pretty(o): None for o in opts}}
        out.append(row("clinc", "choice", ex["text"], q, onehot([pretty(o) for o in opts], pretty(gold)), "eval" if i < n_eval else "train", i))
    return out


def src_ag_news(n, n_eval):
    ds = load_dataset("fancyzhx/ag_news", split="train")
    names = {0: "world", 1: "sports", 2: "business", 3: "science and technology"}
    desc = {"world": "international and political news", "sports": "sports and athletics", "business": "markets, companies, economy", "science and technology": "science, tech, gadgets"}
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        gold = names[ex["label"]]
        opts = subset_options(list(names.values()), gold, k_max=4)
        q = {"type": "choice", "instructions": paraphrase(["What is the topic of this article?", "Which section does this news belong to?"]), "criteria": {o: (desc[o] if rng.random() < 0.6 else None) for o in opts}}
        out.append(row("ag_news", "choice", ex["text"], q, onehot(opts, gold), "eval" if i < n_eval else "train", i))
    return out


def src_dbpedia(n, n_eval):
    ds = load_dataset("fancyzhx/dbpedia_14", split="train")
    names = [pretty(x) for x in ds.features["label"].names]
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        gold = names[ex["label"]]
        opts = subset_options(names, gold, k_max=8)
        q = {"type": "choice", "instructions": paraphrase(["What kind of entity does this text describe?", "Which category is the subject of this article?"]), "criteria": {o: None for o in opts}}
        out.append(row("dbpedia", "choice", ex["content"], q, onehot(opts, gold), "eval" if i < n_eval else "train", i))
    return out


def src_emotion(n, n_eval):
    ds = load_dataset("dair-ai/emotion", "split", split="train")
    names = ds.features["label"].names
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        gold = names[ex["label"]]
        opts = subset_options(names, gold, k_max=6)
        q = {"type": "choice", "instructions": paraphrase(["What emotion does the writer express?", "Which emotion best describes this text?"]), "criteria": {o: None for o in opts}}
        out.append(row("emotion", "choice", ex["text"], q, onehot(opts, gold), "eval" if i < n_eval else "train", i))
    return out


def src_tweet_sentiment(n, n_eval):
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
    levels = ["negative", "neutral", "positive"]  # ordinal -> score
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        q = {"type": "score", "instructions": paraphrase(["What is the sentiment of this tweet?", "Rate the sentiment expressed here."]), "criteria": levels}
        out.append(row("tweet_sentiment", "score", ex["text"], q, onehot(["0", "1", "2"], str(ex["label"])), "eval" if i < n_eval else "train", i))
    return out


def src_sst5(n, n_eval):
    ds = load_dataset("SetFit/sst5", split="train")
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        q = {"type": "score", "instructions": paraphrase(["How positive is this movie review?", "Rate the sentiment of this review."]), "criteria": levels}
        out.append(row("sst5", "score", ex["text"], q, onehot([str(j) for j in range(5)], str(ex["label"])), "eval" if i < n_eval else "train", i))
    return out


def src_yelp(n, n_eval):
    ds = load_dataset("Yelp/yelp_review_full", split="train")
    levels = ["1 star - terrible", "2 stars - poor", "3 stars - okay", "4 stars - good", "5 stars - excellent"]
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        q = {"type": "score", "instructions": paraphrase(["How many stars did this reviewer give?", "Rate this review's overall satisfaction."]), "criteria": levels}
        out.append(row("yelp", "score", ex["text"][:3000], q, onehot([str(j) for j in range(5)], str(ex["label"])), "eval" if i < n_eval else "train", i))
    return out


def src_boolq(n, n_eval):
    ds = load_dataset("google/boolq", split="train")
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        state = {"passage": ex["passage"], "question": ex["question"]} if rng.random() < 0.5 else f"{ex['passage']}\n\nQuestion: {ex['question']}"
        q = {"type": "noul", "instructions": paraphrase(["Based on the passage, is the answer to the question yes?", "Does the passage support answering the question with 'yes'?"])}
        out.append(row("boolq", "noul", state, q, {"yes": 1.0 if ex["answer"] else 0.0, "no": 0.0 if ex["answer"] else 1.0}, "eval" if i < n_eval else "train", i))
    return out


def src_mnli(n, n_eval):
    ds = load_dataset("nyu-mll/glue", "mnli", split="train")
    names = ["entailment", "neutral", "contradiction"]
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        state = {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
        if rng.random() < 0.5:
            q = {"type": "choice", "instructions": "What is the relationship between the premise and the hypothesis?", "criteria": {"entailment": "the hypothesis must be true given the premise", "neutral": "the premise neither confirms nor denies the hypothesis", "contradiction": "the hypothesis cannot be true given the premise"}}
            out.append(row("mnli", "choice", state, q, onehot(names, names[ex["label"]]), "eval" if i < n_eval else "train", i))
        else:
            q = {"type": "noul", "instructions": "Does the premise entail the hypothesis?"}
            yes = 1.0 if ex["label"] == 0 else 0.0
            out.append(row("mnli", "noul", state, q, {"yes": yes, "no": 1 - yes}, "eval" if i < n_eval else "train", i))
    return out


def src_irony(n, n_eval):
    ds = load_dataset("cardiffnlp/tweet_eval", "irony", split="train")
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        q = {"type": "noul", "instructions": paraphrase(["Is this tweet ironic?", "Is the author being ironic or sarcastic?"])}
        out.append(row("irony", "noul", ex["text"], q, {"yes": float(ex["label"]), "no": 1 - float(ex["label"])}, "eval" if i < n_eval else "train", i))
    return out


def src_toxicchat(n, n_eval):
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    out = []
    for i, ex in enumerate(take(ds, n + n_eval)):
        q = {"type": "noul", "instructions": paraphrase(["Is this user message toxic?", "Should this message be flagged as toxic content?"]), "criteria": {"true": "harassment, hate, sexual content, violence or other toxic content", "false": "benign request or conversation"}}
        out.append(row("toxicchat", "noul", ex["user_input"][:4000], q, {"yes": float(ex["toxicity"]), "no": 1 - float(ex["toxicity"])}, "eval" if i < n_eval else "train", i))
    return out


def src_hate_speech(n, n_eval):
    """Multi-annotator: per-comment distribution over {not hateful, unclear, hateful} (0/1/2 per rater)."""
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    by = collections.defaultdict(list)
    text = {}
    for ex in ds:
        by[ex["comment_id"]].append(int(ex["hatespeech"]))
        text[ex["comment_id"]] = ex["text"]
    ids = [c for c, v in by.items() if len(v) >= 3]
    rng.shuffle(ids)
    levels = ["not hate speech", "unclear or borderline", "hate speech"]
    out = []
    for i, c in enumerate(ids[: n + n_eval]):
        votes = by[c]
        t = {str(k): votes.count(k) / len(votes) for k in range(3)}
        q = {"type": "score", "instructions": paraphrase(["Is this comment hate speech?", "Rate how clearly this comment constitutes hate speech."]), "criteria": levels}
        out.append(row("hate_speech", "score", text[c], q, t, "eval" if i < n_eval else "train", i))
    return out


def src_go_emotions(n, n_eval):
    """Multi-rater: for each text, distribution over the emotions raters picked (subset of 3-6 options incl. all voted ones)."""
    ds = load_dataset("google-research-datasets/go_emotions", "raw", split="train")
    emo = [c for c in ds.column_names if c not in {"text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"}]
    votes = collections.defaultdict(collections.Counter)
    text = {}
    nrat = collections.Counter()
    for ex in ds:
        if ex["example_very_unclear"]:
            continue
        nrat[ex["id"]] += 1
        text[ex["id"]] = ex["text"]
        for e in emo:
            if ex[e]:
                votes[ex["id"]][e] += 1
    ids = [i for i, k in nrat.items() if k >= 3 and votes[i]]
    rng.shuffle(ids)
    out = []
    for i, tid in enumerate(ids[: n + n_eval]):
        v = votes[tid]
        voted = list(v)
        if len(voted) > 6:
            continue
        lo = max(2, len(voted))
        k = rng.randint(lo, min(8, lo + 3))
        others = [e for e in emo if e not in v]
        rng.shuffle(others)
        opts = voted + others[: k - len(voted)]
        rng.shuffle(opts)
        total = sum(v.values())
        t = {o: v.get(o, 0) / total for o in opts}
        q = {"type": "choice", "instructions": paraphrase(["Which emotion is most strongly expressed?", "What emotion does this comment convey?"]), "criteria": {o: None for o in opts}}
        out.append(row("go_emotions", "choice", text[tid], q, t, "eval" if i < n_eval else "train", i))
    return out


# ----------------------------------------------------------------------------- constructed long states (truth by construction)
BILLING = ["I was charged twice this month.", "My invoice shows the wrong amount.", "Can I get a refund for last month?", "The payment failed but money left my account.", "Why did my subscription price go up?"]
TECHNICAL = ["The API returns 500 on every request.", "The app crashes when I open settings.", "Exports have been failing since yesterday.", "Login times out after the latest update.", "Webhooks stopped firing this morning."]
SALES = ["What does the enterprise plan cost?", "Can we get a quote for 50 seats?", "Do you offer annual discounts?", "I'd like a demo of the Pro features.", "Is there a startup pricing tier?"]
URGENT = ["This is blocking our launch today.", "We are losing money every minute this is down.", "URGENT - production is affected.", "Need this fixed ASAP."]
CALM = ["No rush on this.", "Whenever you get a chance.", "Just curious, low priority.", "Thanks in advance!"]
PLANS = ["free", "starter", "pro", "enterprise"]
REGIONS = ["us-east", "us-west", "eu-central", "ap-south"]


def _filler(noise_pool: list[str], n_tokens_approx: int) -> list[str]:
    out, n = [], 0
    while n < n_tokens_approx:
        s = rng.choice(noise_pool)
        out.append(s)
        n += len(s.split()) * 1.3
    return out


def src_synthetic(n, n_eval, noise_pool, long=False):
    out = []
    lengths = [8000, 12000, 16000, 24000] if long else [300, 800, 1500, 3000, 5000]
    name = "synthetic_long" if long else "synthetic"
    for i in range(n + n_eval):
        split = "eval" if i < n_eval else "train"
        target_len = rng.choice(lengths)
        kind = rng.choice(["thread", "thread", "record"])
        if kind == "thread":
            # message counts per category -> proportional target (controlled ambiguity)
            counts = {"billing": 0, "technical": 0, "sales": 0}
            mode = rng.random()
            if mode < 0.4:      # clear
                counts[rng.choice(list(counts))] = rng.randint(2, 4)
            elif mode < 0.8:    # two-way split
                a, b = rng.sample(list(counts), 2)
                counts[a], counts[b] = rng.randint(1, 3), rng.randint(1, 3)
            else:               # three-way
                for k in counts:
                    counts[k] = rng.randint(1, 2)
            urgent_n = rng.choice([0, 0, 1, 2, 3])
            msgs = ([("billing", s) for s in rng.sample(BILLING, counts["billing"])] + [("technical", s) for s in rng.sample(TECHNICAL, counts["technical"])] + [("sales", s) for s in rng.sample(SALES, counts["sales"])])
            rng.shuffle(msgs)
            lines = [f"[customer] {m}" for _, m in msgs]
            lines += [f"[customer] {u}" for u in rng.sample(URGENT, urgent_n)]
            if urgent_n == 0:
                lines.append(f"[customer] {rng.choice(CALM)}")
            history = [f"[history] {s}" for s in _filler(noise_pool, target_len)]
            rng.shuffle(lines)
            state = "\n".join(history + ["", "=== CURRENT THREAD ==="] + lines)
            total = sum(counts.values())
            dept_target = {k: v / total for k, v in counts.items()}
            qtype = rng.choice(["dept", "urgent", "refund"])
            if qtype == "dept":
                q = {"type": "choice", "instructions": "Which team should handle the CURRENT THREAD? If the thread mixes topics, weigh them by how much of the thread each takes up.", "criteria": {"billing": "payments, invoices, refunds", "technical": "bugs, outages, integrations", "sales": "pricing, plans, quotes"}}
                out.append(row(name, "choice", state, q, dept_target, split, i))
            elif qtype == "urgent":
                levels = ["not urgent", "somewhat urgent", "urgent", "critical"]
                q = {"type": "score", "instructions": "How urgent is the CURRENT THREAD?", "criteria": levels}
                out.append(row(name, "score", state, q, onehot(["0", "1", "2", "3"], str(min(urgent_n, 3))), split, i))
            else:
                asked = any("refund" in m for _, m in msgs)
                q = {"type": "noul", "instructions": "Does the customer ask for a refund in the CURRENT THREAD?"}
                out.append(row(name, "noul", state, q, {"yes": float(asked), "no": 1 - float(asked)}, split, i))
        else:
            rec = {"account_id": f"acct_{rng.randint(10000, 99999)}", "plan": rng.choice(PLANS), "region": rng.choice(REGIONS), "seats": rng.randint(1, 500), "past_due": rng.random() < 0.3, "mrr_usd": rng.randint(0, 20000), "notes": " ".join(_filler(noise_pool, target_len))}
            if rng.random() < 0.3:
                del rec["plan"]  # teach abstention: 'not mentioned'
            state = rec
            qtype = rng.choice(["plan", "past_due", "seats"])
            if qtype == "plan":
                opts = PLANS + ["not mentioned"]
                rng.shuffle(opts)
                q = {"type": "choice", "instructions": "Which plan is this account on?", "criteria": {o: None for o in opts}}
                out.append(row(name, "choice", state, q, onehot(opts, rec.get("plan", "not mentioned")), split, i))
            elif qtype == "past_due":
                q = {"type": "noul", "instructions": "Is this account past due?"}
                out.append(row(name, "noul", state, q, {"yes": float(rec["past_due"]), "no": 1 - float(rec["past_due"])}, split, i))
            else:
                levels = ["1-10 seats", "11-50 seats", "51-200 seats", "more than 200 seats"]
                lvl = 0 if rec["seats"] <= 10 else 1 if rec["seats"] <= 50 else 2 if rec["seats"] <= 200 else 3
                q = {"type": "score", "instructions": "How large is this account by seat count?", "criteria": levels}
                out.append(row(name, "score", state, q, onehot(["0", "1", "2", "3"], str(lvl)), split, i))
    return out


SOURCES = {
    "banking77": src_banking77, "clinc": src_clinc, "ag_news": src_ag_news, "dbpedia": src_dbpedia, "emotion": src_emotion,
    "tweet_sentiment": src_tweet_sentiment, "sst5": src_sst5, "yelp": src_yelp,
    "boolq": src_boolq, "mnli": src_mnli, "irony": src_irony, "toxicchat": src_toxicchat,
    "hate_speech": src_hate_speech, "go_emotions": src_go_emotions,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jevify_calib.jsonl")
    ap.add_argument("--per-source", type=int, default=3000)
    ap.add_argument("--eval-per-source", type=int, default=300)
    ap.add_argument("--synthetic", type=int, default=4000)
    ap.add_argument("--synthetic-long", type=int, default=1000, help="constructed 8k-24k token states")
    ap.add_argument("--only", default=None, help="comma-separated subset of sources")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    wanted = args.only.split(",") if args.only else list(SOURCES) + ["synthetic", "synthetic_long"]
    noise_pool = [ex["text"] for ex in take(load_dataset("fancyzhx/ag_news", split="train"), 2000, seed=1)]
    for name in wanted:
        try:
            if name == "synthetic":
                got = src_synthetic(args.synthetic, args.eval_per_source, noise_pool)
            elif name == "synthetic_long":
                got = src_synthetic(args.synthetic_long, 150, noise_pool, long=True)
            else:
                got = SOURCES[name](args.per_source, args.eval_per_source)
            rows.extend(got)
            print(f"{name:16} {len(got):6d} rows")
        except Exception as e:  # keep going; report at the end
            print(f"{name:16} FAILED: {e}")
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = collections.Counter((r["source"], r["split"]) for r in rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    for (s, sp), c in sorted(by.items()):
        print(f"  {s:16} {sp:5} {c}")


if __name__ == "__main__":
    main()
