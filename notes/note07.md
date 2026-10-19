# Card history in Redis, decision log in Postgres

Code: `api/app/history.py`, `api/app/store.py`, new endpoints in `api/app/main.py`.

## Card history (Redis)
Each payment's card is identified by `card1|card2|addr1`. One Redis pipeline (one round trip)
reads and updates:

| Signal | Meaning |
|---|---|
| `tx_last_hour` | Payments by this card in the last hour, including this one |
| `avg_amount` | The card's average amount before this payment |
| `amount_vs_avg` | This amount divided by that average (e.g. 10.0 = ten times usual) |
| `new_device` | Device not seen on this card before |

These are recorded and returned with each decision. They are **not** model inputs: the champion
was trained without them, so using them would require retraining.

## Decision log (Postgres)
- Table `decisions`: time, transaction, amount, product, card, fraud score, decision, threshold,
  latency, card-history signals, and a `label` (known outcome, replayed data only, never used for
  the decision).
- A background thread writes in batches, so saving never delays the answer.
- The API waits for the database on startup and keeps answering even if Redis or Postgres fail.
- New endpoints: `GET /stats` (optionally `?minutes=N`) and `GET /decisions/recent`.

## Results (2,000 replay payments: 1,000 fraud, 1,000 genuine, via `/stats`)
| Metric | Value |
|---|---|
| Frauds caught | **49.7%** (497/1,000) |
| Genuine wrongly flagged | **1.1%** (11/1,000) |
| Decision time incl. Redis | **0.26 ms** average, **0.45 ms** p99 |
| Fraud stopped | $59,344 |
| Fraud missed | $83,418 |

- Rates match notebook 05 on the full replay month (48.9% caught, 1.32% false alarms).
- **By value, only 42% of fraud dollars are stopped**, against ~50% of fraud payments: the frauds the
  model misses are larger on average than the ones it catches.

## Tests
14 automated tests: card history logic (fakeredis), decision log against a real Postgres,
plus the earlier API tests.
