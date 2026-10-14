# Day 1 — 14 Oct

Ran `01_explore_and_split.ipynb` on Kaggle against the IEEE-CIS Fraud Detection data.

## What the data looks like
- 590,540 payments, 434 columns, covering about 6 months (days 1–182).
- Fraud rate: **3.50%** (20,663 frauds).
- A model that never flags fraud would still be 96.5% "accurate", so accuracy is
  useless here. I'll judge models on PR-AUC, recall at a fixed false-alarm rate,
  and cost instead.

## Fraud over time
- The first ~4 weeks are much busier (up to ~37k payments/week vs ~20k later),
  possibly a holiday shopping period.
- The weekly fraud rate moves between ~2% and ~5%, lowest in the busy weeks,
  with peaks around weeks 8 and 16.
- Because fraud patterns shift over time, I split by time instead of randomly,
  so models are always tested on payments that came after their training data.

## Missing values
- 214 of 434 columns are more than half empty; 12 are over 90% empty
  (mostly identity/device fields that only exist for some payments).
- Tree models handle missing values, so I'm keeping them for now.

## Leakage check
- Scored every column on its own (AUC). The best was `C4` at 0.68.
- No single column gives the answer away, so no obvious leak.
- Fraud only shows in combinations of signals, which is what tree ensembles are good at.
- Excluded from model features: `TransactionID`, `TransactionDT`, `day`, `week`
  (they only record order/time).

## Splits
| Split  | Days    | Payments | Fraud rate |
|--------|---------|----------|------------|
| Train  | 1–120   | 414,542  | 3.52%      |
| Valid  | 121–152 | 90,568   | 3.39%      |
| Replay | 153–182 | 85,430   | 3.50%      |

Fraud rates are similar across splits, so later tests should be fair.

## Next
Day 2: start the model contest with a single decision tree, then prune it.