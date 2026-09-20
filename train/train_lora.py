"""LoRA calibration training: make the model's first-token label distribution match ground-truth targets.

Loss per item = KL(target || softmax(label_logits)) + mass_weight * (-log P(any label token))
  * label_logits[j] = logsumexp over the token variants of label j ("A", " A") - identical to jevify's scoring
  * the mass term keeps ~100% of next-token probability on the label tokens (format never drifts)

Prompts are jevify's `render()` output through the model's chat template, so training == inference.

usage (one GPU):
  CUDA_VISIBLE_DEVICES=4 uv run --extra train python train/train_lora.py \
      --model google/gemma-4-26B-A4B-it --data data/jevify_calib.jsonl --out runs/g26 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevify.prompting import render  # noqa: E402
from jevify.questions import question_from_dict  # noqa: E402

LOG = None


def log(msg, **kv):
    line = {"t": round(time.time(), 1), "msg": msg, **kv}
    print(json.dumps(line), flush=True)
    if LOG:
        LOG.write(json.dumps(line) + "\n")
        LOG.flush()


# ----------------------------------------------------------------------------- data
def label_variant_ids(tok, label: str) -> list[int]:
    ids = []
    for v in (label, " " + label, label.lower(), " " + label.lower()):  # same variants jevify matches at inference
        enc = tok.encode(v, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def build_examples(tok, rows, max_len, no_think=True):
    """-> list of dicts {ids, variants (list[list[int]]), target (list[float]), source}"""
    kw = {"enable_thinking": False} if no_think else {}
    texts, meta = [], []
    for r in rows:
        q = question_from_dict(r["question"])
        rd = render(r["state"], q)
        t = [r["target"].get(k, 0.0) for k in rd.keys]
        s = sum(t)
        if s <= 0:
            continue
        texts.append(tok.apply_chat_template(rd.messages, tokenize=False, add_generation_prompt=True, **kw))
        meta.append((rd.labels, [x / s for x in t], r["source"]))
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    cache = {}
    out, skipped = [], 0
    for ids, (labels, target, source) in zip(enc, meta):
        if len(ids) > max_len:
            skipped += 1
            continue
        variants = []
        for lab in labels:
            if lab not in cache:
                cache[lab] = label_variant_ids(tok, lab)
            variants.append(cache[lab])
        out.append({"ids": ids, "variants": variants, "target": target, "source": source})
    return out, skipped


def make_batches(examples, micro_tokens, shuffle, seed):
    """Length-bucketed micro-batches with sum(padded_len) <= micro_tokens."""
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in idx:
        L = len(examples[i]["ids"])
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > micro_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def collate(examples, batch_idx, pad_id, device):
    exs = [examples[i] for i in batch_idx]
    L = max(len(e["ids"]) for e in exs)
    ids = torch.full((len(exs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(exs), L), dtype=torch.long)
    for r, e in enumerate(exs):  # right pad; we gather each row's last real position below
        n = len(e["ids"])
        ids[r, :n] = torch.tensor(e["ids"])
        mask[r, :n] = 1
    return ids.to(device), mask.to(device), exs


def last_token_logits(model, ids, mask):
    """Run the decoder, gather the hidden state at each row's last real token, apply lm_head.

    Avoids (a) left-padding, whose fully-masked query rows make SDPA produce NaNs, and
    (b) materializing (B, L, vocab) logits for every position.
    """
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    # No attention mask on purpose: trailing pads are ordinary tokens that causal attention keeps
    # invisible to earlier positions, and we only read the last *real* position. Masking them
    # instead creates fully-masked query rows (pads beyond the sliding window) -> NaN -> NaN grads.
    h = core.model(input_ids=ids).last_hidden_state
    last = mask.sum(-1) - 1
    hl = h[torch.arange(h.size(0), device=h.device), last]
    logits = core.lm_head(hl).float()
    cap = getattr(core.config.get_text_config(), "final_logit_softcapping", None)
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


def label_logits_from_vocab(logits_row: torch.Tensor, variants: list[list[int]]) -> torch.Tensor:
    return torch.stack([torch.logsumexp(logits_row[v], dim=0) for v in variants])


def loss_and_stats(last_logits: torch.Tensor, exs, mass_weight: float):
    """last_logits: (B, V) float32. Returns (loss, stats dict)."""
    total, kl_sum, mass_sum = 0.0, 0.0, 0.0
    correct = 0
    confs = []
    for row, e in zip(last_logits, exs):
        ll = label_logits_from_vocab(row, e["variants"])  # (K,)
        logq = torch.log_softmax(ll, dim=0)
        t = torch.tensor(e["target"], device=row.device)
        nz = t > 0
        kl = (t[nz] * (torch.log(t[nz]) - logq[nz])).sum()
        log_mass = torch.logsumexp(ll, 0) - torch.logsumexp(row, 0)
        total = total + kl - mass_weight * log_mass
        kl_sum += float(kl.detach())
        mass_sum += float(log_mass.detach().exp())
        correct += int(logq.argmax() == t.argmax())
        confs.append(float(logq.max().exp()))
    B = len(exs)
    return total / B, {"kl": kl_sum / B, "mass": mass_sum / B, "acc": correct / B, "conf": float(np.mean(confs))}


@torch.no_grad()
def evaluate(model, examples, pad_id, device, micro_tokens):
    model.eval()
    P, T, M = [], [], []
    for b in make_batches(examples, micro_tokens, shuffle=False, seed=0):
        ids, mask, exs = collate(examples, b, pad_id, device)
        last = last_token_logits(model, ids, mask)
        for row, e in zip(last, exs):
            ll = label_logits_from_vocab(row, e["variants"])
            P.append(torch.softmax(ll, 0).cpu().numpy())
            T.append(np.array(e["target"]))
            M.append(float((torch.logsumexp(ll, 0) - torch.logsumexp(row, 0)).exp()))
    model.train()
    nll = brier = 0.0
    conf, hit = [], []
    for p, t in zip(P, T):
        p = np.clip(p, 1e-9, 1)
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
    return {"n": n, "nll": nll / n, "brier": brier / n, "ece": float(ece), "acc": float(hit.mean()), "conf": float(conf.mean()), "mass": float(np.mean(M))}


# ----------------------------------------------------------------------------- main
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-regex", default=r".*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)")
    ap.add_argument("--micro-tokens", type=int, default=32768, help="token budget per micro-batch (padded)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--mass-weight", type=float, default=0.25)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=800)
    ap.add_argument("--limit", type=int, default=None, help="use only N training rows (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="also save a merged full model to out/merged")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--experts-impl", default="grouped_mm", help="transformers MoE experts implementation: grouped_mm | batched_mm | eager")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOG = open(out / "log.jsonl", "a")
    log("args", **vars(args))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = [json.loads(l) for l in open(args.data)]
    train_rows = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed).shuffle(eval_rows)
    if args.limit:
        train_rows = train_rows[: args.limit]
    eval_rows = eval_rows[: args.eval_n]
    t0 = time.time()
    train_ex, sk1 = build_examples(tok, train_rows, args.max_len)
    eval_ex, sk2 = build_examples(tok, eval_rows, args.max_len)
    ntok = sum(len(e["ids"]) for e in train_ex)
    log("data", train=len(train_ex), eval=len(eval_ex), skipped_too_long=sk1 + sk2, train_tokens=ntok, tokenize_s=round(time.time() - t0, 1))

    load_kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, experts_implementation=args.experts_impl, **load_kw).to(device)
        log("experts_impl", requested=args.experts_impl, used=getattr(model.config, "_experts_implementation", None))
    except (ValueError, TypeError) as e:  # dense model or unsupported kernel name
        log("experts_impl_fallback", error=str(e)[:200])
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=args.target_regex, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log("model", trainable=n_train, total=n_all, pct=round(100 * n_train / n_all, 3), mem_gb=round(torch.cuda.memory_allocated() / 1e9, 1))
    model.train()

    base_eval = evaluate(model, eval_ex, pad_id, device, args.micro_tokens)
    log("eval", step=0, **base_eval)

    batches_per_epoch = len(make_batches(train_ex, args.micro_tokens, shuffle=False, seed=0))
    total_steps = math.ceil(batches_per_epoch * args.epochs / args.accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    log("plan", micro_batches_per_epoch=batches_per_epoch, optimizer_steps=total_steps)

    step, micro, best = 0, 0, float("inf")
    t_start = time.time()
    tokens_seen = 0
    epoch = 0
    done = False
    while not done:
        for b in make_batches(train_ex, args.micro_tokens, shuffle=True, seed=args.seed + epoch):
            ids, mask, exs = collate(train_ex, b, pad_id, device)
            loss, st = loss_and_stats(last_token_logits(model, ids, mask), exs, args.mass_weight)
            if not torch.isfinite(loss):
                log("nonfinite_loss", step=step, micro=micro)
                opt.zero_grad(set_to_none=True)
                continue
            (loss / args.accum).backward()
            tokens_seen += int(mask.sum())
            micro += 1
            if micro % args.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if not torch.isfinite(gn):
                    log("nonfinite_grad", step=step, grad_norm=float(gn))
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    el = time.time() - t_start
                    log("train", step=step, loss=round(float(loss), 4), grad_norm=round(float(gn), 3), lr=sched.get_last_lr()[0], tok_per_s=int(tokens_seen / el), eta_min=round((total_steps - step) * el / step / 60, 1), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), **{k: round(v, 4) for k, v in st.items()})
                if step % args.eval_every == 0 or step == total_steps:
                    ev = evaluate(model, eval_ex, pad_id, device, args.micro_tokens)
                    log("eval", step=step, **ev)
                    if ev["nll"] < best:
                        best = ev["nll"]
                        model.save_pretrained(out / "adapter_best")
                        log("saved", which="best", step=step, nll=ev["nll"])
                if step >= total_steps:
                    done = True
                    break
        epoch += 1
        model.save_pretrained(out / f"adapter_epoch{epoch}")

    model.save_pretrained(out / "adapter_final")
    log("done", steps=step, minutes=round((time.time() - t_start) / 60, 1), best_nll=best)
    if args.merge:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, out / "adapter_best").merge_and_unload()
        merged.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")
        try:  # multimodal bases (Gemma 4): keep the image processor so the merged model stays multimodal
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(args.model).save_pretrained(out / "merged")
        except Exception as e:  # text-only base
            log("no_processor", error=str(e)[:120])
        log("merged", path=str(out / "merged"))


if __name__ == "__main__":
    main()
