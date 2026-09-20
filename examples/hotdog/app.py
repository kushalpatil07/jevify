"""Not Hotdog. The SeeFood app from Silicon Valley, on jevify.

One image in, one question: is this a hot dog? Plus what it actually is, in the same call.

    uv run --extra transformers python examples/hotdog/app.py --model kushalpatil/jevify-gemma4-e4b
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from jevify import Choice, Jevify, Noul  # noqa: E402

FOODS = ["hot dog", "corn dog", "sausage without bun", "hamburger", "sub sandwich", "pizza", "burrito", "taco", "sushi", "salad", "ice cream", "banana", "an animal", "not food", "something else"]
QUESTIONS = {
    "hotdog": Noul(
        "Is this a hot dog?",
        {"true": "a sausage served in a sliced hot dog bun, toppings allowed",
         "false": "anything else: a corn dog, a sausage with no bun, a sub or any other sandwich, a burger, or something that is not food"},
    ),
    "food": Choice("What is shown in the image?", {f: None for f in FOODS}),
}

app = FastAPI(title="Not Hotdog")
SAMPLES = Path(__file__).parent / "samples"
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
jev: Jevify | None = None
lock = threading.Lock()


@app.get("/api/samples")
def samples():
    return sorted(p.name for p in SAMPLES.glob("*.jpg"))


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((768, 768))
    t = time.perf_counter()
    with lock:
        res = jev.system_one({"image": img}, QUESTIONS)
    a = res["answers"]
    food = a["food"]
    top = sorted(food["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    return JSONResponse({
        "hotdog": a["hotdog"]["noul"],
        "verdict": "HOTDOG" if a["hotdog"]["noul"] >= 0.5 else "NOT HOTDOG",
        "food": food["choice"],
        "food_top3": top,
        "ms": round((time.perf_counter() - t) * 1000),
        "input_tokens": res["usage"]["input_tokens"],
    })


def main():
    global jev
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kushalpatil/jevify-gemma4-e4b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    jev = Jevify.from_transformers(args.model)
    import uvicorn

    print(f"Not Hotdog -> http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
