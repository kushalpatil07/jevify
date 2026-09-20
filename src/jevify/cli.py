"""`jevify serve` and `jevify ask`."""
from __future__ import annotations

import argparse
import json
import sys

from .backends.openai_compat import RUNTIME_PRESETS
from .engine import Jevify
from .questions import Choice, Noul, Score


def _add_backend_args(p: argparse.ArgumentParser):
    p.add_argument("--model", required=True, help="model name as the runtime knows it (e.g. gemma4:e4b) or an HF repo id for --backend transformers")
    p.add_argument("--runtime", default="ollama", choices=sorted(RUNTIME_PRESETS), help="which local server you're running (sets base-url default)")
    p.add_argument("--base-url", default=None, help="override the OpenAI-compatible base URL, e.g. http://host:11434/v1")
    p.add_argument("--backend", default="openai", choices=["openai", "transformers"], help="'openai' = talk to a running server (default); 'transformers' = load the model in-process")
    p.add_argument("--api-key", default="local")
    p.add_argument("--think", action="store_true", help="leave reasoning/thinking on (default: off)")
    p.add_argument("--extra-body", default=None, help="JSON merged into every chat/completions request")
    p.add_argument("--concurrency", type=int, default=1, help="parallel requests per call for server backends (use >1 for vLLM/SGLang; keep 1 for Ollama/llama.cpp so their prompt cache hits)")


def _build(args) -> Jevify:
    if args.backend == "transformers":
        return Jevify.from_transformers(args.model)
    return Jevify.from_openai(
        model=args.model,
        base_url=args.base_url,
        runtime=args.runtime,
        api_key=args.api_key,
        no_think=not args.think,
        extra_body=json.loads(args.extra_body) if args.extra_body else None,
        concurrency=getattr(args, "concurrency", 1),
    )


def _parse_kv(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"expected 'key: ...', got {spec!r}")
    k, v = spec.split(":", 1)
    return k.strip(), v.strip()


def cmd_ask(args):
    jev = _build(args)
    questions = {}
    for spec in args.noul or []:
        k, v = _parse_kv(spec)
        questions[k] = Noul(v)
    for spec in args.choice or []:
        k, v = _parse_kv(spec)
        opts = [o.strip() for o in v.split(",") if o.strip()]
        questions[k] = Choice(args.choice_instructions or f"Which of the following best describes the state?", {o: None for o in opts})
    for spec in args.score or []:
        k, v = _parse_kv(spec)
        levels = [o.strip() for o in v.split(",") if o.strip()]
        questions[k] = Score(args.score_instructions or f"Rate the state on the following ordered levels.", levels)
    if not questions:
        raise SystemExit("give at least one of --noul / --choice / --score")
    state = args.state
    if state == "-":
        state = sys.stdin.read()
    out = jev.system_one(state, questions, debug=args.debug)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_serve(args):
    import uvicorn

    from .server import create_app

    jev = _build(args)
    print(f"jevify: backend={jev.backend.name} model={jev.backend.model} -> http://{args.host}:{args.port}/v1/systemone", file=sys.stderr)
    uvicorn.run(create_app(jev), host=args.host, port=args.port, log_level="info")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jevify", description="Run a local LLM as a Jev-style probabilistic decision API.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the Jev-compatible HTTP server")
    _add_backend_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = sub.add_parser("ask", help="ask questions from the command line")
    _add_backend_args(a)
    a.add_argument("--state", required=True, help="the state text ('-' to read stdin)")
    a.add_argument("--noul", action="append", metavar="KEY: question", help="binary question")
    a.add_argument("--choice", action="append", metavar="KEY: opt1,opt2,...", help="categorical question")
    a.add_argument("--score", action="append", metavar="KEY: low,...,high", help="ordinal question")
    a.add_argument("--choice-instructions", default=None)
    a.add_argument("--score-instructions", default=None)
    a.add_argument("--debug", action="store_true", help="include label mass and top tokens")
    a.set_defaults(fn=cmd_ask)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
