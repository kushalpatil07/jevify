"""Publish a trained model: merge the LoRA and push merged weights (+ adapter) to Hugging Face.

  uv run --extra train python train/export.py --base google/gemma-4-E4B-it --adapter runs/e4b/adapter_best \
      --repo kushalpatil/jevify-gemma4-e4b --results runs/e4b/ood.txt
"""
from __future__ import annotations

import argparse
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
pip install "jevify[transformers] @ git+https://github.com/kushalpatil07/jevify"
jevify serve --model {repo}                     # POST /v1/systemone, Jev wire format
```

```python
from jevify import Jevify, Noul, Choice, Score
jev = Jevify.from_transformers("{repo}")        # or serve it with vLLM and use Jevify.from_runtime("vllm", "{repo}")
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
    try:  # keep the image processor for multimodal bases
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(a.base).save_pretrained(merged)
    except Exception:
        pass
    results = Path(a.results).read_text() if a.results else "see repo README"
    (merged / "README.md").write_text(CARD.format(base=a.base, repo=a.repo, results=f"```\n{results}\n```"))
    api = HfApi()
    api.create_repo(a.repo, exist_ok=True, private=a.private)
    api.upload_folder(folder_path=str(merged), repo_id=a.repo, commit_message="merged LoRA")
    if a.adapter:
        api.create_repo(a.repo + "-lora", exist_ok=True, private=a.private)
        api.upload_folder(folder_path=a.adapter, repo_id=a.repo + "-lora", commit_message="LoRA adapter")
    print(f"pushed https://huggingface.co/{a.repo}", file=sys.stderr)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--merged", default=None, help="merged dir (default: <adapter>/../merged; created if missing)")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. kushalpatil/jevify-gemma4-e4b")
    p.add_argument("--results", default=None, help="text file pasted into the model card")
    p.add_argument("--private", action="store_true")
    cmd_push(p.parse_args())


if __name__ == "__main__":
    main()
