"""Qualitative multimodal probe: real photos + human captions + nuanced jevify questions.
Prints the caption and the model's answers so a human can judge whether they make sense.

  CUDA_VISIBLE_DEVICES=6 uv run --extra train python train/probe_image.py --model google/gemma-4-26B-A4B-it --adapter runs/g26_snap --n 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevify.prompting import SYSTEM_PROMPT, render  # noqa: E402
from jevify.questions import Choice, Noul, Score  # noqa: E402
from jevify.scoring import match_label_probs, normalize  # noqa: E402

QUESTIONS = {
    "subject": Choice("What is the main subject of the photo?", {"person or people": None, "animal": None, "vehicle": None, "food": None, "landscape or nature": None, "building or street": None, "object": None}),
    "outdoors": Noul("Was this photo taken outdoors?"),
    "people": Score("How many people are visible?", ["none", "one", "two or three", "four or more"]),
    "smiling": Noul("Is anyone in the photo clearly smiling?"),
    "water": Noul("Is there water (sea, lake, river, pool) visible?"),
    "mood": Score("What is the mood of the scene?", ["calm or neutral", "playful or joyful", "tense or dramatic"]),
    "dog": Noul("Is there a dog in the photo?"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)
    it = iter(ds)
    for i in range(args.n):
        ex = next(it)
        img = ex["image"].convert("RGB")
        cap = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
        print(f"\n#{i+1}  CAPTION: {cap}")
        for key, q in QUESTIONS.items():
            r = render("[the attached image]", q, SYSTEM_PROMPT)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": r.messages[0]["content"]}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": r.messages[1]["content"]}]},
            ]
            inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
            with torch.inference_mode():
                lp = torch.log_softmax(model(**inputs, logits_to_keep=1).logits[0, -1].float(), -1)
            top = torch.topk(lp, 64)
            table = {proc.tokenizer.decode([int(t)]): float(v) for v, t in zip(top.values, top.indices)}
            raw, mass = match_label_probs(table, r.labels)
            probs = normalize(raw)
            p = {k: probs[lab] for lab, k in zip(r.labels, r.keys)}
            if isinstance(q, Noul):
                print(f"    {key:9} P(yes)={p['yes']:.2f}")
            elif isinstance(q, Score):
                ev = sum(int(k) * v for k, v in p.items())
                dist = " ".join(f"{q.criteria[int(k)]}:{v:.2f}" for k, v in p.items())
                print(f"    {key:9} score={ev:.2f}  [{dist}]")
            else:
                top3 = sorted(p.items(), key=lambda kv: -kv[1])[:3]
                print(f"    {key:9} " + "  ".join(f"{k}:{v:.2f}" for k, v in top3))


if __name__ == "__main__":
    main()
