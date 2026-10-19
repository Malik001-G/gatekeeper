# Replay simulator

Code: `simulator/` — a small service (port 8001) that streams the held-out month into the API.

## How it works
- Payments are sent in their original time order, each with its own dataset time
  (`2017-12-01 + TransactionDT`), so card history sees realistic gaps between a card's payments.
- Each payment carries its known outcome (`label`), stored for reporting only.
- Speed is set in payments per second (`rate`; 0 = as fast as possible), with optional parallel
  senders (`workers`) for load tests.
- Control endpoints: `POST /start`, `POST /stop`, `GET /status`, `POST /reset-history`
  (clears card history in Redis). A second start while one is running is refused.
- Command-line runner for load tests: `python -m sim.cli`.

## API hardening
- `/stats`, `/decisions/recent` and `/reset-history` return a clear 503 when Postgres or Redis is
  down, instead of an internal error.
- With Redis down, `/score` keeps answering (card history empty) and `/health` shows `"redis": false`.

## Live replay (rate 50/s, first 537 payments)
- Rate held at exactly 50.0/s.
- 4/10 frauds caught, 5 false alarms (0.95%).

## Load test (20,000 payments in time order, 8 parallel senders, MacBook Pro)
| Metric | Value |
|---|---|
| Throughput | **1,354 payments/second**, 0 errors |
| Frauds caught | **48.1%** (292/607) |
| Genuine wrongly flagged | **1.11%** (216/19,393) |
| Fraud rate in stream | 3.0% |

The stream's catch rate and false alarm rate match notebook 05 on the full replay month
(48.9%, 1.32%): the same result now holds in the notebook, through the API on balanced samples,
and through the live stream in real order.

At ~1,350 payments/second, one container would handle over 100 million payments a day
(on this hardware).

## Tests
6 simulator tests: payload format, time order, limits and rate, stop, error counting, parallel
senders. 19 automated tests across the API and simulator in total.

