"""Send real held-out payments to the running API and compare its decisions with the truth.

Usage (inside the api container):
    python scripts/score_samples.py --n 20
    python scripts/score_samples.py --n 50 --fraud 10     # include 10 known frauds
    python scripts/score_samples.py --n 1 --show-json      # print one request body
    python scripts/score_samples.py --n 2000 --fraud 1000 --quiet   # summary only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request

import pandas as pd

NOT_FEATURES = {"isFraud", "TransactionID", "TransactionDT", "day", "week"}


def to_payload(row: pd.Series) -> dict:
    features = {}
    for k, v in row.items():
        if k in NOT_FEATURES:
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue  # missing fields are simply left out
        features[k] = v.item() if hasattr(v, "item") else v
    return {"transaction_id": str(int(row["TransactionID"])), "features": features}


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of payments to send")
    ap.add_argument("--fraud", type=int, default=0, help="how many of them should be known frauds")
    ap.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--data", default=os.path.join(os.environ.get("DATA_DIR", "/data"), "replay.parquet"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--show-json", action="store_true", help="print the first request body")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    frauds = df[df["isFraud"] == 1].sample(min(args.fraud, args.n), random_state=args.seed)
    rest = df[df["isFraud"] == 0].sample(args.n - len(frauds), random_state=args.seed)
    batch = pd.concat([frauds, rest]).sample(frac=1, random_state=args.seed)

    rows, correct, caught, false_alarms, lat = [], 0, 0, 0, []
    for _, row in batch.iterrows():
        body = to_payload(row)
        if args.show_json and not rows:
            print(json.dumps(body, indent=2)[:3000], "\n")
        res = post(f"{args.url}/score", body)
        truth = "fraud" if row["isFraud"] == 1 else "genuine"
        right = (res["decision"] == "suspicious") == (truth == "fraud")
        correct += right
        caught += (truth == "fraud") and res["decision"] == "suspicious"
        false_alarms += (truth == "genuine") and res["decision"] == "suspicious"
        lat.append(res["latency_ms"])
        rows.append((body["transaction_id"], f"${row['TransactionAmt']:.2f}", res["decision"],
                     f"{res['fraud_score']:.3f}", truth, "✓" if right else "✗", f"{res['latency_ms']:.1f}"))

    if not args.quiet:
        header = ("transaction", "amount", "decision", "score", "truth", "", "ms")
        widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
        for r in [header] + rows:
            print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
        print()
    n_fraud, n_genuine = len(frauds), len(rows) - len(frauds)
    print(f"{correct}/{len(rows)} decisions matched the truth")
    if n_fraud:
        print(f"Frauds caught:          {caught}/{n_fraud} ({caught / n_fraud:.1%})")
    if n_genuine:
        print(f"Genuine wrongly flagged: {false_alarms}/{n_genuine} ({false_alarms / n_genuine:.2%})")
    lat = sorted(lat)
    print(f"Decision time: median {lat[len(lat) // 2]:.2f} ms, max {lat[-1]:.2f} ms")


if __name__ == "__main__":
    main()