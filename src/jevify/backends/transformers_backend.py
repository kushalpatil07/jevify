"""In-process backend on HF transformers: exact full-vocab logits, no server needed.

Two ways to answer N questions about one state, both prefill the state exactly once:

* `next_token(messages)` — llama.cpp-style prompt cache: the KV cache of the previous prompt is
  kept and, if the new prompt shares a token prefix with it, only the differing suffix is run.

* `next_token_batch(list_of_messages)` — all questions in **one forward pass**. The shared token
  prefix P is (re)used from the cache, then one of two ways to run the question suffixes S_i:

  - batched branches (default): copy P's KV cache N times along the batch dimension and run the
    N suffixes as N rows. Plain causal masks, so it is exact for every architecture (including
    sliding-window models like Gemma). Costs N copies of the prefix KV, so it is used while
    that fits in `max_branch_kv_gb`.
  - packed tree attention (`packed=True`, full-attention models only): concatenate the suffixes
    into a single row with a 4D mask so S_i attends to P + itself, never S_j. One shared copy
    of the prefix, so it scales to very long states.

        kv:      P P P P | S1 S1 | S2 S2 S2 | S3
        S1 rows: 1 1 1 1 | causal|   0      | 0
        S2 rows: 1 1 1 1 |   0   | causal   | 0
        S3 rows: 1 1 1 1 |   0   |   0      | causal

Install with `pip install jevify[transformers]`.
"""
from __future__ import annotations

from typing import Any

from ..images import load_image
from .base import NextToken


def _has_images(messages) -> bool:
    return any(isinstance(m.get("content"), list) for m in messages)


def supports_packed(config) -> bool:
    """The packed tree mask replaces every layer's mask, which is only exact when all layers use
    full causal attention. Sliding-window models (Gemma) would let local layers see the whole
    prefix, diverging from what llama.cpp/Ollama compute -> use the sequential path for them.
    TODO: pass a per-layer-type mask dict ({"full_attention": ..., "sliding_attention": ...}).
    """
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    layer_types = getattr(text, "layer_types", None)
    if layer_types and any(t != "full_attention" for t in layer_types):
        return False
    return getattr(text, "sliding_window", None) in (None, 0)


def _common_prefix_len(seqs: list[list[int]]) -> int:
    if not seqs:
        return 0
    n = min(len(s) for s in seqs)
    first = seqs[0]
    for i in range(n):
        t = first[i]
        for s in seqs[1:]:
            if s[i] != t:
                return i
    return n


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        model: str,
        device: str | None = None,
        dtype: Any = "auto",
        top_k: int = 64,
        no_think: bool = True,
        min_shared_prefix: int = 16,
        prefill_chunk: int = 2048,
        packed: bool | None = None,
        max_branch_kv_gb: float = 24.0,
        **from_pretrained_kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model
        self.model = model
        self.top_k = top_k
        self.no_think = no_think
        self.min_shared_prefix = min_shared_prefix
        self.prefill_chunk = prefill_chunk
        self.max_branch_kv_gb = max_branch_kv_gb
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._processor = None  # loaded on first image request (multimodal bases only)
        from_pretrained_kwargs.setdefault("attn_implementation", "sdpa")  # custom 4D bool masks need sdpa/eager
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **from_pretrained_kwargs).to(self.device).eval()
        # tree-mask packing is only exact on full-attention models; batched branches work everywhere
        self.packed = supports_packed(self.lm.config) if packed is None else bool(packed)
        self._prefix_ids: list[int] = []  # tokens currently held in self._cache
        self._cache = None

    # -- images: go through the model's own processor, no prefix cache ----------------
    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            try:
                self._processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception as e:  # text-only model
                raise RuntimeError(f"{self.model_id} has no image processor; images need a multimodal base (e.g. Gemma 4)") from e
        return self._processor

    def _next_token_multimodal(self, messages) -> NextToken:
        torch = self.torch
        mm = []
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                c = [({"type": "image", "image": load_image(b["image"])} if b.get("type") == "image" else b) for b in c]
            else:
                c = [{"type": "text", "text": c}]
            mm.append({"role": m["role"], "content": c})
        kw = {"enable_thinking": False} if self.no_think else {}
        inputs = self.processor.apply_chat_template(mm, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kw).to(self.device)
        with torch.inference_mode():
            logits = self.lm(**inputs, logits_to_keep=1).logits[0, -1].float()
            table = self._table(torch.log_softmax(logits, dim=-1))
        return NextToken(logprobs=table, prompt_tokens=int(inputs["input_ids"].shape[1]), cached_tokens=0, extra={"device": self.device, "mode": "multimodal"})

    # -- prompt rendering -------------------------------------------------------------
    def encode(self, messages: list[dict[str, str]]) -> list[int]:
        kw = {"enable_thinking": False} if self.no_think else {}
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        text = "\n\n".join(m["content"] for m in messages) + "\nAnswer:"
        return self.tokenizer(text)["input_ids"]

    # -- cache management -------------------------------------------------------------
    def _cache_len(self) -> int:
        return self._cache.get_seq_length() if self._cache is not None else 0

    def _ensure_prefix(self, prefix: list[int]) -> None:
        """Make self._cache hold exactly `prefix` (reusing whatever it already shares)."""
        from transformers import DynamicCache

        shared = _common_prefix_len([self._prefix_ids, prefix]) if self._cache is not None else 0
        if shared < self.min_shared_prefix:
            self._cache, self._prefix_ids, shared = DynamicCache(), [], 0
        elif self._cache_len() > shared:
            self._cache.crop(shared - self._cache_len())  # negative = remove that many tokens
        self._prefix_ids = self._prefix_ids[:shared]
        todo = prefix[shared:]
        for i in range(0, len(todo), self.prefill_chunk):  # chunk long states to bound activation memory
            chunk = todo[i : i + self.prefill_chunk]
            inp = self.torch.tensor([chunk], device=self.device)
            out = self.lm(input_ids=inp, past_key_values=self._cache, use_cache=True, logits_to_keep=1)
            self._cache = out.past_key_values
            self._prefix_ids.extend(chunk)

    def _table(self, logprobs_1d) -> dict[str, float]:
        top = self.torch.topk(logprobs_1d, k=min(self.top_k, logprobs_1d.numel()))
        return {self.tokenizer.decode([int(i)]): float(lp) for lp, i in zip(top.values.tolist(), top.indices.tolist())}

    # -- single question (sequential prompt-cache path) -------------------------------
    def next_token(self, messages: list[dict[str, str]]) -> NextToken:
        if _has_images(messages):
            return self._next_token_multimodal(messages)
        torch = self.torch
        from transformers import DynamicCache

        ids = self.encode(messages)
        with torch.inference_mode():
            # reuse whatever prefix the cache already holds, then run the rest in ONE forward
            shared = _common_prefix_len([self._prefix_ids, ids]) if self._cache is not None else 0
            shared = min(shared, len(ids) - 1)
            if shared < self.min_shared_prefix:
                self._cache, shared = DynamicCache(), 0
            elif self._cache_len() > shared:
                self._cache.crop(shared - self._cache_len())
            todo = ids[shared:]
            for i in range(0, len(todo), self.prefill_chunk):
                inp = torch.tensor([todo[i : i + self.prefill_chunk]], device=self.device)
                out = self.lm(input_ids=inp, past_key_values=self._cache, use_cache=True, logits_to_keep=1)
                self._cache = out.past_key_values
            self._prefix_ids = list(ids)
            table = self._table(torch.log_softmax(out.logits[0, -1].float(), dim=-1))
        return NextToken(logprobs=table, prompt_tokens=len(ids), cached_tokens=shared, extra={"device": self.device, "mode": "sequential"})

    # -- all questions in ONE forward pass --------------------------------------------
    def _prefix_kv_bytes(self) -> int:
        total = 0
        for layer in getattr(self._cache, "layers", []):
            for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if t is not None and hasattr(t, "numel"):
                    total += t.numel() * t.element_size()
        return total

    def next_token_batch(self, batch: list[list[dict[str, str]]]) -> list[NextToken]:
        if len(batch) == 1 or any(_has_images(m) for m in batch):
            return [self.next_token(m) for m in batch]
        if self.packed:
            return self._next_token_batch_packed(batch)
        return self._next_token_batch_branches(batch)

    def _next_token_batch_branches(self, batch: list[list[dict[str, str]]]) -> list[NextToken]:
        """Prefill the shared prefix once, copy its KV N times, run the N question suffixes as a batch."""
        import copy

        torch = self.torch
        ids = [self.encode(m) for m in batch]
        p_len = min(_common_prefix_len(ids), min(len(s) for s in ids) - 1)
        prefix = ids[0][:p_len]
        suffixes = [s[p_len:] for s in ids]
        n = len(batch)
        with torch.inference_mode():
            self._ensure_prefix(prefix)
            if n * self._prefix_kv_bytes() > self.max_branch_kv_gb * 1e9:  # too big to copy -> one at a time
                return [self.next_token(m) for m in batch]
            L = max(len(s) for s in suffixes)
            pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            inp = torch.full((n, L), pad, dtype=torch.long)
            mask = torch.zeros((n, p_len + L), dtype=torch.long)
            mask[:, :p_len] = 1
            for r, s_ in enumerate(suffixes):  # right pad; gather each row's last real token below
                inp[r, : len(s_)] = torch.tensor(s_)
                mask[r, p_len : p_len + len(s_)] = 1
            cache = copy.deepcopy(self._cache)  # keep the clean prefix cache for the next call
            cache.batch_repeat_interleave(n)
            out = self.lm(input_ids=inp.to(self.device), attention_mask=mask.to(self.device), past_key_values=cache, use_cache=True)
            last = torch.tensor([len(s_) - 1 for s_ in suffixes], device=self.device)
            logits = out.logits[torch.arange(n, device=self.device), last].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            tables = [self._table(logprobs[i]) for i in range(n)]
            del cache, out
        return [
            NextToken(logprobs=t, prompt_tokens=len(s), cached_tokens=p_len, extra={"device": self.device, "mode": "branches"})
            for t, s in zip(tables, ids)
        ]

    def _next_token_batch_packed(self, batch: list[list[dict[str, str]]]) -> list[NextToken]:
        torch = self.torch
        ids = [self.encode(m) for m in batch]
        p_len = min(_common_prefix_len(ids), min(len(s) for s in ids) - 1)  # every branch keeps >= 1 token
        prefix = ids[0][:p_len]
        suffixes = [s[p_len:] for s in ids]
        lens = [len(s) for s in suffixes]
        q_len = sum(lens)

        with torch.inference_mode():
            self._ensure_prefix(prefix)
            cached_before = len(self._prefix_ids)  # == p_len after _ensure_prefix
            kv_len = p_len + q_len

            # tree mask: (1, 1, q_len, kv_len), True = may attend
            mask = torch.zeros(q_len, kv_len, dtype=torch.bool)
            mask[:, :p_len] = True
            pos = torch.empty(q_len, dtype=torch.long)
            last_idx = []
            start = 0
            for n in lens:
                blk = slice(start, start + n)
                mask[blk, p_len + start : p_len + start + n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
                pos[blk] = torch.arange(p_len, p_len + n)  # every branch restarts at |P|
                last_idx.append(start + n - 1)
                start += n
            packed = torch.tensor([sum(suffixes, [])], device=self.device)
            out = self.lm(
                input_ids=packed,
                attention_mask=mask[None, None].to(self.device),
                position_ids=pos[None].to(self.device),
                cache_position=torch.arange(p_len, kv_len, device=self.device),
                past_key_values=self._cache,
                use_cache=True,
                logits_to_keep=torch.tensor(last_idx, device=self.device),
            )
            logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)  # (n_questions, vocab)
            tables = [self._table(logprobs[i]) for i in range(len(batch))]
            # drop the branch KV so the cache holds the clean shared prefix for the next call
            self._cache = out.past_key_values
            self._cache.crop(-q_len)
            self._prefix_ids = prefix

        return [
            NextToken(logprobs=t, prompt_tokens=len(s), cached_tokens=cached_before, extra={"device": self.device, "mode": "packed"})
            for t, s in zip(tables, ids)
        ]
