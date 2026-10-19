"""Run a replay from the command line and print progress (useful for load tests).

    python -m sim.cli --rate 50 --limit 2000
    python -m sim.cli --rate 0 --limit 20000 --workers 8     # as fast as possible
"""
from __future__ import annotations

import argparse
import os
import time

import httpx

from .replayer import Replayer, load_replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=20, help="payments per second (0 = as fast as possible)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--url", default=os.environ.get("API_URL", "http://api:8000"))
    ap.add_argument("--data", default=os.path.join(os.environ.get("DATA_DIR", "/data"), "replay.parquet"))
    args = ap.parse_args()

    client = httpx.Client(base_url=args.url, timeout=10,
                          limits=httpx.Limits(max_connections=32, max_keepalive_connections=32))

    def send(payload: dict) -> dict:
        r = client.post("/score", json=payload)
        r.raise_for_status()
        return r.json()

    rep = Replayer(load_replay(args.data), send)
    rep.start(rate=args.rate, limit=args.limit, start=args.start, workers=args.workers)
    while rep.running:
        time.sleep(2)
        p = rep.progress.as_dict()
        print(f"{p['sent']:>7,}/{p['total']:,} sent · {p['rate_actual']:>6.1f}/s · flagged {p['flagged']:,} · "
              f"caught {p['frauds_caught']}/{p['frauds']} · errors {p['errors']}", flush=True)
    p = rep.progress.as_dict()
    print("\nDone.")
    for k in ["sent", "errors", "elapsed_s", "rate_actual", "flagged", "frauds", "frauds_caught",
              "catch_rate", "false_alarms", "false_alarm_rate", "payment_time", "last_error"]:
        print(f"  {k:<17} {p[k]}")
    client.close()


if __name__ == "__main__":
    main()
