"""Jev-compatible HTTP server: POST /v1/systemone.

Drop-in for the official SDK:
    TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=local
"""
from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .engine import Jevify
from .questions import question_from_dict


class SystemOneRequest(BaseModel):
    state: Any
    model: str | None = None  # accepted for wire compatibility; the server decides the model
    questions: dict[str, dict] = Field(min_length=1)


def create_app(jev: Jevify) -> FastAPI:
    app = FastAPI(title="jevify", version="0.1.0")
    lock = threading.Lock()  # one local model; serialize so the prompt cache stays warm per call

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": jev.backend.name, "model": jev.backend.model}

    @app.post("/v1/systemone")
    def system_one(req: SystemOneRequest, debug: bool = Query(False)):
        try:
            questions = {k: question_from_dict(q) for k, q in req.questions.items()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            with lock:
                return jev.system_one(req.state, questions, debug=debug)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

    return app
