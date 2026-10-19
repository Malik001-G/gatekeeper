"""Live card history in Redis: what this card has been doing recently.

These signals are recorded with every decision and shown on the dashboard. They are NOT
model inputs: the champion was trained without them, so feeding them in would need retraining.

One Redis round trip per payment (a pipeline):
  - payments by this card in the last hour (sorted set of timestamps)
  - this card's average amount before this payment (running sum and count)
  - whether this device has been seen on this card before (set)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import redis

KEEP_SECONDS = 7 * 24 * 3600   # forget a card after a week of silence
WINDOW_SECONDS = 3600          # "last hour"


@dataclass
class CardSignals:
    card_key: Optional[str]
    tx_last_hour: Optional[int] = None
    avg_amount: Optional[float] = None
    amount_vs_avg: Optional[float] = None
    new_device: Optional[bool] = None

    def as_dict(self) -> dict:
        return asdict(self)


def card_key(features: dict) -> Optional[str]:
    """A card is identified by its card number group, bank code and billing area."""
    parts = [features.get("card1"), features.get("card2"), features.get("addr1")]
    if parts[0] is None:
        return None
    return "|".join("" if p is None else str(int(p)) if isinstance(p, float) and p == int(p) else str(p)
                    for p in parts)


class CardHistory:
    def __init__(self, url: str):
        self.r = redis.Redis.from_url(url, socket_timeout=0.2, socket_connect_timeout=0.5)

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except redis.RedisError:
            return False

    def observe(self, features: dict, event_ts: Optional[float] = None) -> CardSignals:
        """Read this card's history, then add the current payment to it."""
        key = card_key(features)
        if key is None:
            return CardSignals(card_key=None)
        ts = event_ts if event_ts is not None else time.time()
        amount = _num(features.get("TransactionAmt"))
        device = features.get("DeviceInfo")

        k_times, k_amt, k_dev = f"card:{key}:times", f"card:{key}:amount", f"card:{key}:devices"
        p = self.r.pipeline(transaction=False)
        p.zremrangebyscore(k_times, "-inf", ts - WINDOW_SECONDS)       # 0
        p.zadd(k_times, {f"{ts}:{uuid.uuid4().hex[:8]}": ts})          # 1
        p.zcount(k_times, ts - WINDOW_SECONDS, ts)                     # 2
        p.hmget(k_amt, "sum", "count")                                 # 3 (before this payment)
        if amount is not None:
            p.hincrbyfloat(k_amt, "sum", amount)
            p.hincrby(k_amt, "count", 1)
        if device:
            p.sismember(k_dev, device)                                 # read before adding
            p.sadd(k_dev, device)
        for k in (k_times, k_amt, k_dev):
            p.expire(k, KEEP_SECONDS)
        res = p.execute()

        s, c = res[3]
        prev_avg = float(s) / int(c) if s is not None and c is not None and int(c) > 0 else None
        seen_device = None
        if device:
            seen_device = bool(res[4 + (2 if amount is not None else 0)])
        return CardSignals(
            card_key=key,
            tx_last_hour=int(res[2]),
            avg_amount=round(prev_avg, 2) if prev_avg is not None else None,
            amount_vs_avg=round(amount / prev_avg, 2) if (prev_avg and amount is not None) else None,
            new_device=(not seen_device) if device else None,
        )


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None
