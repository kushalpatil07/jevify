"""Image references in state: a PIL image, a file path, an http(s) URL, or a data URL / base64 string."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any


def is_image_ref(x: Any) -> bool:
    if x.__class__.__name__ == "Image" or hasattr(x, "convert") and hasattr(x, "size"):
        return True
    if isinstance(x, str):
        s = x.strip()
        return s.startswith(("http://", "https://", "data:image/")) or (len(s) < 1024 and Path(s).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"} and Path(s).exists())
    return False


def load_image(ref: Any):
    """-> PIL.Image (RGB). Needs pillow."""
    from PIL import Image

    if hasattr(ref, "convert"):
        return ref.convert("RGB")
    s = str(ref).strip()
    if s.startswith("data:image/"):
        return Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1]))).convert("RGB")
    if s.startswith(("http://", "https://")):
        import httpx

        return Image.open(io.BytesIO(httpx.get(s, follow_redirects=True, timeout=60).content)).convert("RGB")
    return Image.open(s).convert("RGB")


def to_data_url(ref: Any) -> str:
    """For OpenAI-compatible servers: pass URLs through, encode everything else as a PNG data URL."""
    if isinstance(ref, str) and ref.strip().startswith(("http://", "https://", "data:image/")):
        return ref.strip()
    img = load_image(ref)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
