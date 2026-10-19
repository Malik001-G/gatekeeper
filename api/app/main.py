"""Gatekeeper scoring API: allow or suspicious, in milliseconds."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from psycopg.types.json import Jsonb

from .history import CardHistory, CardSignals, card_key
from .model import Champion

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gatekeeper")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
REDIS_URL = os.environ.get("REDIS_URL")
DATABASE_URL = os.environ.get("DATABASE_URL")
# Safety rule in front of the model: amounts it was never trained on are not left to the model.
MAX_AMOUNT = float(os.environ.get("MAX_AMOUNT", "50000"))

state: dict = {"champion": None, "error": None, "history": None, "store": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["champion"] = Champion(MODEL_DIR)
    except Exception as e:  # keep the API up so /health can report the problem
        state["error"] = str(e)
    if REDIS_URL:
        state["history"] = CardHistory(REDIS_URL)
    if DATABASE_URL:
        from .store import DecisionStore
        state["store"] = DecisionStore(DATABASE_URL)
    yield
    if state["store"]:
        state["store"].close()


app = FastAPI(
    title="Gatekeeper",
    description="Real-time fraud scoring: allow or suspicious, with a fraud score, the top 3 reasons "
                "and the card's recent history.",
    version="0.4.0",
    lifespan=lifespan,
)

Value = Optional[Union[float, int, str]]


class Payment(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Your reference for this payment")
    features: dict[str, Value] = Field(
        ...,
        description="Payment fields by column name, e.g. TransactionAmt, ProductCD, card1. "
                    "Missing fields are treated as unknown.",
        examples=[{"TransactionAmt": 117.0, "ProductCD": "W", "card1": 9500, "card2": 111,
                   "addr1": 204, "P_emaildomain": "gmail.com"}],
    )
    event_time: Optional[datetime] = Field(None, description="When the payment happened (default: now)")
    source: Optional[str] = Field(None, description="Where the payment came from, e.g. simulator or manual")
    label: Optional[Literal[0, 1]] = Field(
        None, description="Known outcome (1 = fraud), for demos with replayed data. "
                          "Stored for reporting only; never used for the decision.")


class History(BaseModel):
    card_key: Optional[str]
    tx_last_hour: Optional[int]
    avg_amount: Optional[float]
    amount_vs_avg: Optional[float]
    new_device: Optional[bool]


class Reason(BaseModel):
    feature: str
    name: str = Field(..., description="Plain-language name of the feature")
    value: str = Field(..., description="The value this payment had")
    effect: str = Field(..., description="raises risk or lowers risk")
    impact: float = Field(..., description="SHAP value: how far it moved the score (log-odds)")
    text: str


class ScoreResult(BaseModel):
    transaction_id: Optional[str]
    decision: str = Field(..., description="allow or suspicious")
    fraud_score: float = Field(..., description="0 to 1; higher means more likely fraud")
    threshold: float
    latency_ms: float = Field(..., description="Total time to decide, including card history")
    model_ms: float
    features_received: int
    rule: Optional[str] = Field(None, description="Set when a safety rule, not the model, decided")
    reasons: Optional[list[Reason]] = Field(
        None, description="Top 3 features behind the decision (SHAP). Present for flagged payments, "
                          "or for every payment with explain=all")
    explain_ms: Optional[float] = None
    card_history: Optional[History]


def _champion() -> Champion:
    if state["champion"] is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {state['error']}")
    return state["champion"]


def _store():
    if state["store"] is None:
        raise HTTPException(status_code=503, detail="Decision log not configured (DATABASE_URL)")
    return state["store"]


@app.get("/health")
def health() -> dict:
    ok = state["champion"] is not None
    hist, store = state["history"], state["store"]
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "error": state["error"],
        "redis": None if hist is None else hist.ping(),
        "decision_log": None if store is None else store.status(),
    }


@app.get("/model")
def model_info() -> dict:
    return _champion().summary()


@app.post("/score", response_model=ScoreResult)
def score(payment: Payment,
          explain: Literal["flagged", "all", "none"] = Query(
              "flagged", description="When to add the top 3 reasons: flagged payments only (default), "
                                     "every payment, or never")) -> ScoreResult:
    t0 = time.perf_counter()
    champ = _champion()
    d = champ.score(payment.features)
    rule = amount_rule(payment.features.get("TransactionAmt"))

    signals = None
    if rule:  # keep impossible amounts out of the card's history
        signals = CardSignals(card_key=card_key(payment.features)) if state["history"] is not None else None
    elif state["history"] is not None:
        ts = payment.event_time.timestamp() if payment.event_time else None
        try:
            signals = state["history"].observe(payment.features, ts)
        except Exception as e:  # history is a bonus; never block a decision on it
            log.warning("card history unavailable: %s", e)
            signals = CardSignals(card_key=card_key(payment.features))

    decision = "suspicious" if (d.suspicious or rule) else "allow"
    reasons, explain_ms = None, None
    if explain == "all" or (explain == "flagged" and decision == "suspicious"):
        reasons, explain_ms = champ.explain(payment.features, suspicious=d.suspicious)
    total_ms = (time.perf_counter() - t0) * 1000

    if state["store"] is not None:
        f = payment.features
        state["store"].add({
            "event_time": payment.event_time or datetime.now(timezone.utc),
            "transaction_id": payment.transaction_id, "source": payment.source,
            "amount": _num(f.get("TransactionAmt")), "product": f.get("ProductCD"),
            "card_key": signals.card_key if signals else card_key(f),
            "fraud_score": d.fraud_score, "decision": decision, "threshold": champ.threshold,
            "latency_ms": total_ms,
            "card_tx_last_hour": signals.tx_last_hour if signals else None,
            "card_avg_amount": signals.avg_amount if signals else None,
            "amount_vs_card_avg": signals.amount_vs_avg if signals else None,
            "new_device": signals.new_device if signals else None,
            "label": payment.label,
            "rule": rule,
            "reasons": Jsonb(reasons) if reasons else None,
        })

    return ScoreResult(
        transaction_id=payment.transaction_id,
        decision=decision,
        fraud_score=round(d.fraud_score, 6),
        threshold=champ.threshold,
        latency_ms=round(total_ms, 2),
        model_ms=round(d.latency_ms, 2),
        features_received=len(payment.features),
        rule=rule,
        reasons=reasons,
        explain_ms=round(explain_ms, 2) if explain_ms is not None else None,
        card_history=History(**signals.as_dict()) if signals else None,
    )


@app.get("/decisions/recent")
def recent_decisions(limit: int = Query(50, ge=1, le=1000),
                     decision: Optional[Literal["allow", "suspicious"]] = None) -> list[dict]:
    try:
        return _store().recent(limit, decision)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Decision log unavailable: {e}")


@app.get("/stats")
def stats(minutes: Optional[int] = Query(None, ge=1, description="Only the last N minutes")) -> dict:
    try:
        return _store().stats(minutes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Decision log unavailable: {e}")


def amount_rule(amount) -> Optional[str]:
    """Flag amounts outside what the model can judge: missing ones are left to the model."""
    a = _num(amount)
    if a is None:
        return None
    if a <= 0:
        return "amount must be positive"
    if a > MAX_AMOUNT:
        return f"amount above ${MAX_AMOUNT:,.0f}: outside the range the model was trained on"
    return None


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None
