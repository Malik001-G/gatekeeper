"""Gatekeeper scoring API.

Day 1: health check only. The /score endpoint arrives on 20 Oct,
once the model contest has picked a champion.
"""
from fastapi import FastAPI

app = FastAPI(
    title="Gatekeeper",
    description="Real-time fraud scoring: allow or suspicious, with reasons.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": False}
