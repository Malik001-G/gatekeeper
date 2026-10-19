"""Control service for the replay: start, stop, status. Used by the dashboard's replay button."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .replayer import Replayer, load_replay

API_URL = os.environ.get("API_URL", "http://api:8000")
DATA_PATH = os.path.join(os.environ.get("DATA_DIR", "/data"), "replay.parquet")
REDIS_URL = os.environ.get("REDIS_URL")

state: dict = {"replayer": None, "error": None, "client": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.Client(base_url=API_URL, timeout=10,
                          limits=httpx.Limits(max_connections=32, max_keepalive_connections=32))
    state["client"] = client

    def send(payload: dict) -> dict:
        r = client.post("/score", json=payload)
        r.raise_for_status()
        return r.json()

    try:
        state["replayer"] = Replayer(load_replay(DATA_PATH), send)
    except Exception as e:
        state["error"] = str(e)
    yield
    if state["replayer"]:
        state["replayer"].stop()
    client.close()


app = FastAPI(title="Gatekeeper simulator", version="0.1.0", lifespan=lifespan)


class StartRequest(BaseModel):
    rate: float = Field(20, ge=0, description="Payments per second; 0 = as fast as possible")
    limit: Optional[int] = Field(None, ge=1, description="How many payments to send (default: all)")
    start: int = Field(0, ge=0, description="Row of the replay month to start from")
    workers: int = Field(1, ge=1, le=32, description="Parallel senders (for load tests)")


def _replayer() -> Replayer:
    if state["replayer"] is None:
        raise HTTPException(503, f"Replay data not loaded: {state['error']}")
    return state["replayer"]


@app.get("/health")
def health() -> dict:
    rep = state["replayer"]
    return {"status": "ok" if rep else "degraded", "payments_available": len(rep.df) if rep else 0,
            "error": state["error"]}


@app.get("/status")
def status() -> dict:
    return _replayer().progress.as_dict()


@app.post("/start")
def start(req: StartRequest) -> dict:
    rep = _replayer()
    try:
        rep.start(rate=req.rate, limit=req.limit, start=req.start, workers=req.workers)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return rep.progress.as_dict()


@app.post("/stop")
def stop() -> dict:
    rep = _replayer()
    rep.stop()
    return rep.progress.as_dict()


@app.post("/reset-history")
def reset_history() -> dict:
    """Forget all card history in Redis, so a new replay starts from a clean slate."""
    if not REDIS_URL:
        raise HTTPException(503, "REDIS_URL not configured")
    if _replayer().running:
        raise HTTPException(409, "stop the replay first")
    r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
    deleted, batch = 0, []
    try:
        for key in r.scan_iter("card:*", count=1000):
            batch.append(key)
            if len(batch) >= 1000:
                deleted += r.delete(*batch)
                batch = []
        if batch:
            deleted += r.delete(*batch)
    except redis.RedisError as e:
        raise HTTPException(503, f"Redis unavailable: {e}")
    return {"deleted_keys": deleted}
