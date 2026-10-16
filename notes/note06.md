# The scoring API

 `api/` (FastAPI, Docker). The champion files from notebook 05 are mounted from `models/`,
the replay month from `data/`. Neither is committed.

## Endpoints
| Endpoint | What it does |
|---|---|
| `POST /score` | One payment in, `allow` / `suspicious` out, with fraud score, threshold and decision time |
| `GET /health` | Whether the model loaded (and why not, if it failed) |
| `GET /model` | Champion details: trees, features, threshold, replay results |
| `GET /docs` | Interactive API docs |

Request body: `{"transaction_id": "...", "features": {"TransactionAmt": 117.0, "ProductCD": "W", ...}}`.
Fields left out are treated as missing, exactly as in training. Unknown fields are ignored.

## Same decisions as the notebook
- Single payments are prepared with a dedicated fast path (`vector()`); batches use the notebook's
  `prepare()`. A test checks both give identical model inputs.
- First version built a pandas table per payment: ~9 ms. Fast path: under 1 ms.

## Test on real held-out payments
`scripts/score_samples.py` sends replay-month payments (never seen in training) through HTTP
and compares each decision with the truth.

30 payments (10 fraud):
- 3/10 frauds caught, 0/20 false alarms. Small sample; 3 or fewer catches happens ~1 time in 6
  at a true 49% rate.
- Missed frauds scored 0.008–0.069, far below the 0.162 threshold: they look entirely genuine
  to the model, not near misses.

2,000 payments (1,000 fraud, 1,000 genuine):

| Metric | API | Notebook 05 (full replay) |
|---|---|---|
| Frauds caught | **50.0%** (500/1,000) | 48.9% |
| Genuine wrongly flagged | **1.70%** (17/1,000) | 1.32% |
| Decision time (median) | **0.11 ms** | 2.5 ms (prepare + predict in pandas) |

Both rates are within sampling noise of the notebook (expected ~13 false alarms per 1,000,
give or take ~4), so the API reproduces the notebook's decisions.

## Tests
7 automated tests (`api/tests/`) using a small stand-in model: health, model info, suspicious and
normal payments, missing and unknown fields, bad numbers, and fast-path parity with `prepare()`.
