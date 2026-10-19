"""Replays the held-out month into the scoring API as a live payment stream.

Payments are sent in their original time order. Each carries:
  - event_time: the payment's own time in the dataset (TransactionDT seconds after 2017-12-01),
    so card history in Redis sees realistic gaps between a card's payments;
  - label: the known outcome, stored for reporting only (the API never uses it to decide);
  - source: "simulator".
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

DATASET_START = datetime(2017, 12, 1, tzinfo=timezone.utc)
NOT_FEATURES = {"isFraud", "TransactionID", "TransactionDT", "day", "week"}


def load_replay(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values("TransactionDT").reset_index(drop=True)


def to_payload(row: dict) -> dict:
    features = {}
    for k, v in row.items():
        if k in NOT_FEATURES or v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        features[k] = v.item() if hasattr(v, "item") else v
    ts = DATASET_START + timedelta(seconds=float(row["TransactionDT"]))
    return {"transaction_id": str(int(row["TransactionID"])), "features": features,
            "event_time": ts.isoformat(), "source": "simulator", "label": int(row["isFraud"])}


@dataclass
class Progress:
    running: bool = False
    total: int = 0
    sent: int = 0
    errors: int = 0
    flagged: int = 0
    frauds: int = 0
    frauds_caught: int = 0
    false_alarms: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    payment_time: Optional[str] = None
    last_error: Optional[str] = None
    rate_target: float = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, payload: dict, result: Optional[dict], error: Optional[str]) -> None:
        with self._lock:
            if error:
                self.errors += 1
                self.last_error = error
                return
            self.sent += 1
            self.payment_time = payload["event_time"]
            suspicious = result["decision"] == "suspicious"
            fraud = payload["label"] == 1
            self.flagged += suspicious
            self.frauds += fraud
            self.frauds_caught += fraud and suspicious
            self.false_alarms += (not fraud) and suspicious

    def as_dict(self) -> dict:
        with self._lock:
            elapsed = ((self.finished_at or time.time()) - self.started_at) if self.started_at else 0
            genuine = self.sent - self.frauds
            return {
                "running": self.running, "total": self.total, "sent": self.sent, "errors": self.errors,
                "progress": round(self.sent / self.total, 4) if self.total else 0,
                "elapsed_s": round(elapsed, 1),
                "rate_actual": round(self.sent / elapsed, 1) if elapsed > 0 else 0,
                "rate_target": self.rate_target,
                "payment_time": self.payment_time,
                "flagged": self.flagged, "frauds": self.frauds, "frauds_caught": self.frauds_caught,
                "false_alarms": self.false_alarms,
                "catch_rate": round(self.frauds_caught / self.frauds, 4) if self.frauds else None,
                "false_alarm_rate": round(self.false_alarms / genuine, 4) if genuine else None,
                "last_error": self.last_error,
            }


class Replayer:
    """Sends rows [start, start+limit) of the replay month at `rate` payments/second (0 = flat out)."""

    def __init__(self, df: pd.DataFrame, send: Callable[[dict], dict]):
        self.df = df
        self.send = send
        self.progress = Progress()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, rate: float = 20, limit: Optional[int] = None, start: int = 0, workers: int = 1) -> None:
        if self.running:
            raise RuntimeError("a replay is already running")
        end = len(self.df) if not limit else min(len(self.df), start + limit)
        self.progress = Progress(running=True, total=max(end - start, 0), started_at=time.time(),
                                 rate_target=rate)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(rate, start, end, workers), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def wait(self) -> None:
        if self._thread:
            self._thread.join()

    def _one(self, payload: dict) -> None:
        try:
            self.progress.record(payload, self.send(payload), None)
        except Exception as e:
            self.progress.record(payload, None, str(e)[:200])

    def _run(self, rate: float, start: int, end: int, workers: int) -> None:
        interval = 1.0 / rate if rate and rate > 0 else 0.0
        next_at = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            for chunk_start in range(start, end, 1000):
                rows = self.df.iloc[chunk_start:min(chunk_start + 1000, end)].to_dict("records")
                for row in rows:
                    if self._stop.is_set():
                        return
                    if interval:
                        delay = next_at - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                        next_at = max(next_at + interval, time.perf_counter() - 1.0)
                    payload = to_payload(row)
                    if pool:
                        pool.submit(self._one, payload)
                    else:
                        self._one(payload)
        finally:
            if pool:
                pool.shutdown(wait=True)
            self.progress.running = False
            self.progress.finished_at = time.time()
