# The champion's final test

Notebook: `notebooks/05_champion.ipynb`. Inputs: the splits from `01_explore_and_split` and the
model files from `04_boosting`.

## Reloading the model the way the API will
- Loaded `lightgbm_model.txt` + `preprocess.json` with a standalone `prepare()` function.
- 472 trees, 431 features (31 text). Valid PR-AUC **0.6195**, identical to notebook 04,
  so the saved files reproduce the trained model exactly.
- `prepare()` first took ~51 ms per payment at 431 columns (column-by-column pandas work);
  rewritten to work on column blocks: ~2 ms, same output.

## Choosing the threshold (on `valid` only)
| Policy | Threshold | Recall | False alarm rate | Precision | Cost |
|---|---|---|---|---|---|
| **1% false alarms (chosen)** | 0.1623 | 52.8% | 1.00% | 65.0% | $256,208 |
| Lowest cost | 0.0337 | 76.5% | 6.63% | 28.8% | $166,160 |

- The lowest-cost threshold looks cheaper only because a false alarm is assumed to cost $10.
  It flags 1 in 15 genuine customers, and 7 in 10 of its flags are genuine payments.
- The 1% policy matches how the contest was judged and keeps customer friction predictable.
  The lowest-cost threshold is saved in `champion.json` as an alternative.

## Final test: `replay` (days 153–182, never used for training, tuning or threshold)
| Metric | Valid | Replay |
|---|---|---|
| PR-AUC | 0.620 | **0.552** |
| Fraud caught (recall) | 52.8% | **48.9%** |
| Genuine wrongly flagged | 1.00% | **1.32%** |
| Flags that are fraud (precision) | 65.0% | **57.3%** |
| Cost | $256,208 | **$274,475** |
| Cost with no system | $492,574 | $457,436 |

Replay month in plain numbers:
- 85,430 payments, 2,994 fraud.
- 2,554 flagged: 1,463 frauds caught (48.9%), 1,091 genuine customers flagged (1.32%).
- Money lost: $274,475 instead of $457,436, so **$182,961 saved (40%)**.

## Why replay is lower
- `valid` was slightly favourable, since early stopping used it.
- Fraud patterns keep shifting; replay is the furthest month from training.
- The false alarm rate rose from 1.00% to 1.32% because the threshold was fixed on earlier scores.
  **Known limitation:** a live system would need the threshold re-checked periodically.

## Speed of one live decision
Prepare + predict, single thread: **median 2.5 ms, p99 4.2 ms** (target: under 50 ms).

## Packaged for the API
- `lightgbm_model.txt` (6.6 MB): the model
- `preprocess.json`: feature order and text codes
- `champion.json`: threshold 0.1623, policy, lowest-cost alternative, valid and replay results, latency

## Next
The scoring API: FastAPI + Docker, serving `POST /score` with the champion.