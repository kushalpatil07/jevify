"""Default backend: any OpenAI-compatible chat endpoint that returns logprobs.

Works with Ollama, llama.cpp `llama-server`, LM Studio, vLLM, SGLang, mlx-lm server, …
One request per question: `max_tokens=1, logprobs=true, top_logprobs=20`. The runtime's own
prompt cache handles the shared `<state>` prefix across questions.
"""
from __future__ import annotations

import httpx

from ..images import to_data_url
from .base import NextToken


def _openai_messages(messages):
    """jevify image blocks -> OpenAI `image_url` blocks (data URLs for local images)."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [({"type": "image_url", "image_url": {"url": to_data_url(b["image"])}} if b.get("type") == "image" else b) for b in c]
        out.append({"role": m["role"], "content": c})
    return out

# Everything Ollama accepts today; other servers cap at 20 as well (OpenAI's limit).
MAX_TOP_LOGPROBS = 20

RUNTIME_PRESETS: dict[str, dict] = {
    # base_url and how to switch off "thinking" for reasoning models
    "ollama": {"base_url": "http://localhost:11434/v1", "no_think": {"reasoning_effort": "none"}},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "no_think": {"reasoning_effort": "none"}},
    "llamacpp": {"base_url": "http://localhost:8080/v1", "no_think": {"chat_template_kwargs": {"enable_thinking": False}}},
    "vllm": {"base_url": "http://localhost:8000/v1", "no_think": {"chat_template_kwargs": {"enable_thinking": False}}},
    "sglang": {"base_url": "http://localhost:30000/v1", "no_think": {"chat_template_kwargs": {"enable_thinking": False}}},
    "mlx": {"base_url": "http://localhost:8080/v1", "no_think": {"chat_template_kwargs": {"enable_thinking": False}}},
    "openai": {"base_url": None, "no_think": {}},
}


class OpenAICompatBackend:
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        runtime: str = "openai",
        api_key: str = "local",
        no_think: bool = True,
        top_logprobs: int = MAX_TOP_LOGPROBS,
        extra_body: dict | None = None,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
        concurrency: int = 1,
    ):
        preset = RUNTIME_PRESETS.get(runtime)
        if preset is None:
            raise ValueError(f"unknown runtime {runtime!r}; one of {sorted(RUNTIME_PRESETS)}")
        base_url = base_url or preset["base_url"]
        if not base_url:
            raise ValueError("base_url is required (e.g. http://localhost:11434/v1)")
        self.name = runtime
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.top_logprobs = min(int(top_logprobs), MAX_TOP_LOGPROBS)
        self.extra_body = dict(preset["no_think"]) if no_think else {}
        self.extra_body.update(extra_body or {})
        self.concurrency = max(1, int(concurrency))
        self._client = client or httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def next_token(self, messages: list[dict[str, str]]) -> NextToken:
        body = {
            "model": self.model,
            "messages": _openai_messages(messages),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "stream": False,
            **self.extra_body,
        }
        r = self._client.post(f"{self.base_url}/chat/completions", json=body)
        if r.status_code >= 400:
            hint = ""
            if "context" in r.text and ("exceed" in r.text or "n_ctx" in r.text):
                hint = (" — the state is longer than the runtime's context window. Ollama defaults to 4096: start it with "
                        "OLLAMA_CONTEXT_LENGTH=32768 (or `launchctl setenv OLLAMA_CONTEXT_LENGTH 32768` for the Mac app, then restart it); "
                        "llama.cpp: -c 32768; vLLM: --max-model-len.")
            raise RuntimeError(f"{self.base_url} returned {r.status_code}: {r.text[:500]}{hint}")
        data = r.json()
        try:
            choice = data["choices"][0]
            content = (choice.get("logprobs") or {}).get("content") or []
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected response shape: {str(data)[:500]}") from e
        if not content:
            raise RuntimeError(
                "server returned no logprobs — does this runtime/model support `logprobs`? "
                f"(model={self.model!r}, base_url={self.base_url!r})"
            )
        first = content[0]
        logprobs = {t["token"]: float(t["logprob"]) for t in first.get("top_logprobs", [])}
        logprobs.setdefault(first["token"], float(first["logprob"]))
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return NextToken(
            logprobs=logprobs,
            prompt_tokens=usage.get("prompt_tokens"),
            cached_tokens=cached,
            extra={"sampled": first["token"], "model": data.get("model")},
        )

    def next_token_batch(self, batch: list[list[dict[str, str]]]) -> list[NextToken]:
        """Sequential by default (so single-slot runtimes hit their prompt cache on the shared
        state); `concurrency > 1` fans out for servers with prefix caching (vLLM, SGLang)."""
        if self.concurrency == 1 or len(batch) == 1:
            return [self.next_token(m) for m in batch]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self.next_token, batch))
