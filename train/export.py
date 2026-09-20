"""Publish a trained model: merge the LoRA, push to Hugging Face, and build an Ollama model.

  # merge + push (run on the training box; needs `hf auth login` or HF_TOKEN)
  uv run --extra train python train/export.py push --base google/gemma-4-E4B-it --adapter runs/e4b/adapter_best \
      --repo kushalpatil/jevify-gemma4-e4b --results runs/e4b/ood.txt

  # build + (optionally) push the Ollama model from the merged safetensors
  uv run --extra train python train/export.py ollama --merged runs/e4b/merged --name kushalpatil/jevify-gemma4-e4b [--push]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CARD = """---
license: gemma
base_model: {base}
library_name: transformers
tags: [jevify, classification, calibration, probabilistic, gemma4, lora]
---

# {repo}

`{base}` fine-tuned (LoRA, merged) to give **honest probabilities** when asked typed questions
about a piece of state — the model behind [jevify](https://github.com/kushalpatil07/jevify), a local,
Jev-compatible probabilistic decision API.

Nothing is generated: one prefill, read the next-token distribution over the answer labels, done.

## Use

```bash
ollama pull {repo}
jevify serve --runtime ollama --model {repo}      # POST /v1/systemone
```

```python
from jevify import Jevify, Noul, Choice, Score
jev = Jevify.from_transformers("{repo}")        # or Jevify.from_runtime("ollama", "{repo}")
jev.system_one("Help! My payouts have been failing for 3 days.", {{
    "urgent": Noul("Does this convey urgency?"),
    "team":   Choice("Which team should handle this?", {{"billing": None, "technical": None, "sales": None}}),
    "mood":   Score("How frustrated is the customer?", ["calm", "frustrated", "furious"]),
}})
```

## Training

LoRA r=64 on attention projections, 2 epochs over ~47k (state, question, target-distribution)
items from 16 sources: hard-labeled classification sets with randomized option subsets and
order, multi-annotator sets with real human label distributions, and constructed long states
(up to 24k tokens) with answers known by construction. Loss = KL(target || label distribution).
No teacher model. Recipe: `train/` in the jevify repo.

## Results

{results}
"""


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), file=sys.stderr)
    subprocess.run(cmd, check=True, **kw)


def cmd_push(a):
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = Path(a.merged or (Path(a.adapter).parent / "merged"))
    if not (merged / "config.json").exists():
        print(f"merging {a.adapter} into {a.base} -> {merged}", file=sys.stderr)
        base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
        model.save_pretrained(merged, safe_serialization=True)
        AutoTokenizer.from_pretrained(a.base).save_pretrained(merged)
    results = Path(a.results).read_text() if a.results else "see repo README"
    (merged / "README.md").write_text(CARD.format(base=a.base, repo=a.repo, results=f"```\n{results}\n```"))
    api = HfApi()
    api.create_repo(a.repo, exist_ok=True, private=a.private)
    api.upload_folder(folder_path=str(merged), repo_id=a.repo, commit_message="merged LoRA")
    if a.adapter:
        api.create_repo(a.repo + "-lora", exist_ok=True, private=a.private)
        api.upload_folder(folder_path=a.adapter, repo_id=a.repo + "-lora", commit_message="LoRA adapter")
    print(f"pushed https://huggingface.co/{a.repo}", file=sys.stderr)


def cmd_ollama(a):
    modelfile = Path(a.merged) / "Modelfile"
    modelfile.write_text(f"FROM {Path(a.merged).resolve()}\nPARAMETER num_ctx {a.num_ctx}\nPARAMETER temperature 0\n")
    run(["ollama", "create", a.name, "-f", str(modelfile)])
    if a.push:
        run(["ollama", "push", a.name])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--merged", default=None, help="merged dir (default: <adapter>/../merged; created if missing)")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. kushalpatil/jevify-gemma4-e4b")
    p.add_argument("--results", default=None, help="text file pasted into the model card")
    p.add_argument("--private", action="store_true")
    p.set_defaults(fn=cmd_push)
    o = sub.add_parser("ollama")
    o.add_argument("--merged", required=True)
    o.add_argument("--name", required=True)
    o.add_argument("--num-ctx", type=int, default=32768)
    o.add_argument("--push", action="store_true")
    o.set_defaults(fn=cmd_ollama)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
