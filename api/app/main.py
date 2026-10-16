"""Gatekeeper scoring API: allow or suspicious, in milliseconds."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import Champion

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
state: dict = {"champion": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["champion"] = Champion(MODEL_DIR)
    except Exception as e:  # keep the API up so /health can report the problem
        state["error"] = str(e)
    yield


app = FastAPI(
    title="Gatekeeper",
    description="Real-time fraud scoring: allow or suspicious, with a fraud score.",
    version="0.2.0",
    lifespan=lifespan,
)

Value = Optional[Union[float, int, str]]


class Payment(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Your reference for this payment")
    features: dict[str, Value] = Field(
        ...,
        description="Payment fields by column name, e.g. TransactionAmt, ProductCD, card1. "
                    "Missing fields are treated as unknown.",
        examples=[{"TransactionAmt": 117.0, "ProductCD": "W", "card1": 9500, "P_emaildomain": "gmail.com"}],
    )


class ScoreResult(BaseModel):
    transaction_id: Optional[str]
    decision: str = Field(..., description="allow or suspicious")
    fraud_score: float = Field(..., description="0 to 1; higher means more likely fraud")
    threshold: float
    latency_ms: float
    features_received: int


def _champion() -> Champion:
    if state["champion"] is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {state['error']}")
    return state["champion"]


@app.get("/health")
def health() -> dict:
    ok = state["champion"] is not None
    return {"status": "ok" if ok else "degraded", "model_loaded": ok, "error": state["error"]}


@app.get("/model")
def model_info() -> dict:
    return _champion().summary()


@app.post("/score", response_model=ScoreResult)
def score(payment: Payment) -> ScoreResult:
    champ = _champion()
    d = champ.score(payment.features)
    return ScoreResult(
        transaction_id=payment.transaction_id,
        decision="suspicious" if d.suspicious else "allow",
        fraud_score=round(d.fraud_score, 6),
        threshold=champ.threshold,
        latency_ms=round(d.latency_ms, 2),
        features_received=len(payment.features),
    )
