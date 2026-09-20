"""Go/no-go spike: can an off-the-shelf local model answer Jev-style questions from
next-token logprobs alone (one prefill, zero decode)?

Talks to any OpenAI-compatible server (default: Ollama) with
  max_tokens=1, logprobs=true, top_logprobs=20
and reads the label tokens (A/B/C, Yes/No, 0..9) out of the top-20.

usage: python spike/spike.py [--model gemma4:e4b] [--base-url http://localhost:11434/v1]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request

SYSTEM = "You are a precise classifier. Reply with a single token: only the label of your answer."


# ----------------------------------------------------------------------------- prompts
def render_state(state) -> str:
    if not isinstance(state, str):
        state = json.dumps(state, indent=2, ensure_ascii=False)
    return f"<state>\n{state}\n</state>\n\n"


def choice_prompt(state, instructions, options: dict[str, str | None]):
    letters = [chr(ord("A") + i) for i in range(len(options))]
    lines = []
    for L, (k, desc) in zip(letters, options.items()):
        lines.append(f"{L}. {k}" + (f" — {desc}" if desc else ""))
    user = render_state(state) + f"{instructions}\n\nOptions:\n" + "\n".join(lines)
    user += "\n\nAnswer with the letter of the single best option."
    return user, letters, list(options.keys())


def score_prompt(state, instructions, levels: list[str]):
    letters = [chr(ord("A") + i) for i in range(len(levels))]
    lines = [f"{L}. {lvl}" for L, lvl in zip(letters, levels)]
    user = render_state(state) + f"{instructions}\n\nLevels (ordered from lowest to highest):\n" + "\n".join(lines)
    user += "\n\nAnswer with the letter of the single best level."
    return user, letters, [str(i) for i in range(len(levels))]


def noul_prompt(state, instructions):
    user = render_state(state) + f"{instructions}\n\nAnswer Yes or No."
    return user, ["Yes", "No"], ["yes", "no"]


# ----------------------------------------------------------------------------- backend
def top_logprobs(base_url, model, user, prefill: str | None, no_think: bool):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
        "stream": False,
    }
    if no_think:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    dt = time.perf_counter() - t0
    content = resp["choices"][0]["logprobs"]["content"]
    if not content:
        raise RuntimeError(f"no logprobs in response: {json.dumps(resp)[:400]}")
    tok = content[0]
    dist = {t["token"]: t["logprob"] for t in tok.get("top_logprobs", [])}
    dist.setdefault(tok["token"], tok["logprob"])
    return dist, resp.get("usage", {}).get("prompt_tokens"), dt, tok["token"]


def label_probs(dist: dict[str, float], labels: list[str]):
    """Sum prob of every top-k token whose stripped/lowercased text equals a label."""
    raw = {L: 0.0 for L in labels}
    for tok, lp in dist.items():
        t = tok.strip().lower()
        for L in labels:
            if t == L.lower():
                raw[L] += math.exp(lp)
    mass = sum(raw.values())
    probs = {L: (raw[L] / mass if mass > 0 else float("nan")) for L in labels}
    return probs, mass


def confidence(p: list[float]) -> float:
    k = len(p)
    h = -sum(x * math.log(x) for x in p if x > 0)
    return 1 - h / math.log(k) if k > 1 else 1.0


# ----------------------------------------------------------------------------- cases
JEV_TICKET = "Help! My payouts have been failing for 3 days."
DEPT = {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"}

CASES = [
    # (kind, state, instructions, options/levels, expected)   expected: set of acceptable answers
    ("choice", JEV_TICKET, "Which team should handle this?", DEPT, {"billing", "technical"}),
    ("choice", "I'd like to upgrade to the enterprise plan, what does it cost?", "Which team should handle this?", DEPT, {"sales"}),
    ("choice", "The API returns 500 on every request since this morning.", "Which team should handle this?", DEPT, {"technical"}),
    ("choice", "I was charged twice for my subscription this month.", "Which team should handle this?", DEPT, {"billing"}),
    ("choice", "Bonjour, je voudrais réserver une table pour deux.", "What language is this?", {"english": None, "french": None, "german": None, "spanish": None}, {"french"}),
    ("choice", "This is the best purchase I've made all year.", "What is the sentiment?", {"positive": None, "neutral": None, "negative": None}, {"positive"}),
    ("choice", "The Fed raised interest rates by 25 basis points.", "What is the topic?", {"sports": None, "finance": None, "cooking": None, "technology": None}, {"finance"}),
    ("noul", "URGENT: production is down, we are losing $10k per minute.", "Does this convey urgency?", None, {"yes"}),
    ("noul", "Just wanted to say thanks for the great docs, no rush on anything.", "Does this convey urgency?", None, {"no"}),
    ("noul", "The capital of France is Paris.", "Is this statement true?", None, {"yes"}),
    ("noul", "Water boils at 50°C at sea level.", "Is this statement true?", None, {"no"}),
    ("score", "I HATE THIS. Third time your app deleted my data. Cancel my account NOW.", "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"], {"2"}),
    ("score", "Hi! Quick question — is there a dark mode? Thanks :)", "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"], {"0"}),
    ("score", "Typo in the footer copyright year.", "How severe is this bug?", ["cosmetic", "minor functionality affected", "major functionality affected", "system unusable"], {"0"}),
    ("score", JEV_TICKET, "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"], {"1", "2"}),
]


def run_case(args, kind, state, instr, opts, prefill, reverse=False):
    if kind == "choice":
        items = list(opts.items())
        if reverse:
            items = items[::-1]
        user, labels, keys = choice_prompt(state, instr, dict(items))
    elif kind == "score":
        user, labels, keys = score_prompt(state, instr, opts)  # keys are level indices "0".."N-1"
    else:
        user, labels, keys = noul_prompt(state, instr)
    dist, ntok, dt, sampled = top_logprobs(args.base_url, args.model, user, prefill, not args.think)
    probs, mass = label_probs(dist, labels)
    by_key = {k: probs[L] for L, k in zip(labels, keys)}
    best = max(by_key, key=by_key.get)
    return by_key, mass, best, dt, ntok, sampled, dist


def fmt(d):
    return "{" + ", ".join(f"{k}: {v:.2f}" for k, v in d.items()) + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--think", action="store_true", help="leave thinking on (default: reasoning_effort=none)")
    ap.add_argument("--prefill", default="Answer: ", help="assistant prefill for the second pass ('' to skip)")
    args = ap.parse_args()

    for prefill in (None, args.prefill or None):
        tag = f"prefill={prefill!r}" if prefill else "no prefill"
        print(f"\n===== {args.model} | {tag} =====")
        correct = stable = 0
        masses = []
        for kind, state, instr, opts, expected in CASES:
            by_key, mass, best, dt, ntok, sampled, dist = run_case(args, kind, state, instr, opts, prefill)
            masses.append(mass)
            ok = best in expected
            correct += ok
            extra = ""
            if kind == "choice":
                by_key_r, mass_r, best_r, *_ = run_case(args, kind, state, instr, opts, prefill, reverse=True)
                st = best_r == best
                stable += st
                extra = f" | reversed→{best_r} {'✓' if st else '✗ UNSTABLE'} {fmt(by_key_r)}"
            elif kind == "score":
                exp_val = sum(int(k) * p for k, p in by_key.items())
                extra = f" | score={exp_val:.2f}"
            top3 = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
            print(f"[{kind:6}] {'✓' if ok else '✗'} {best:<10} mass={mass:.3f} conf={confidence(list(by_key.values())):.2f} {dt*1000:5.0f}ms "
                  f"{fmt(by_key)}{extra}\n         state={state[:60]!r} top3={[(t, round(math.exp(lp), 3)) for t, lp in top3]}")
        n_choice = sum(1 for c in CASES if c[0] == "choice")
        print(f"\n--> correct {correct}/{len(CASES)} | order-stable {stable}/{n_choice} | "
              f"label mass min={min(masses):.3f} mean={sum(masses)/len(masses):.3f}")

    # ---- ambiguous cases: no right answer, just look at whether the distribution spreads
    print("\n===== ambiguous cases (calibration probe) =====")
    AMBIG = [
        ("choice", "I want a refund because the app crashed and lost my work.", "Which team should handle this?", DEPT),
        ("choice", "Hey, is the Pro plan going to fix the export bug I reported?", "Which team should handle this?", DEPT),
        ("choice", "It was fine I guess.", "What is the sentiment?", {"positive": None, "neutral": None, "negative": None}),
        ("noul", "The meeting might get moved to tomorrow, not sure yet.", "Is the meeting tomorrow?", None),
        ("score", "Ugh, the export failed again. Can someone look into it when you get a chance?", "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
    ]
    prefill = args.prefill or None
    for kind, state, instr, opts in AMBIG:
        by_key, mass, best, dt, ntok, sampled, dist = run_case(args, kind, state, instr, opts, prefill)
        print(f"[{kind:6}] {best:<10} conf={confidence(list(by_key.values())):.2f} {fmt(by_key)}  state={state[:55]!r}")

    # ---- prompt-cache timing: long shared state, 3 questions back to back
    print("\n===== prompt cache timing (long state, 3 questions) =====")
    filler = ("Ticket history entry: customer contacted support about an unrelated question, resolved. " * 120)
    long_state = filler + "\n\nLATEST MESSAGE: " + JEV_TICKET
    prefill = args.prefill or None
    for i, (kind, instr, opts) in enumerate([
        ("choice", "Which team should handle the LATEST MESSAGE?", DEPT),
        ("noul", "Does the LATEST MESSAGE convey urgency?", None),
        ("score", "How frustrated is the customer in the LATEST MESSAGE?", ["Calm", "Frustrated", "Very angry"]),
    ]):
        by_key, mass, best, dt, ntok, *_ = run_case(args, kind, long_state, instr, opts, prefill)
        print(f"q{i+1} [{kind:6}] {dt*1000:6.0f}ms  prompt_tokens={ntok}  best={best} mass={mass:.3f} {fmt(by_key)}")


if __name__ == "__main__":
    main()
