# Day 1 — Data exploration and time-based splits

Notebook: `notebooks/01_explore_and_split.ipynb`, run on Kaggle against the IEEE-CIS Fraud Detection data.

## The data
- 590,540 payments, 434 columns, covering about 6 months (days 1–182).
- Fraud rate: **3.50%** (20,663 frauds).
- A model that never flags fraud would still be 96.5% "accurate", so accuracy is not used.
  Models are judged on PR-AUC, recall at a fixed false-alarm rate, and cost.

## Fraud over time
- The first ~4 weeks are much busier (up to ~37k payments/week vs ~20k later).
- The weekly fraud rate moves between ~2% and ~5%, lowest in the busy weeks,
  with peaks around weeks 8 and 16.
- Because fraud patterns shift over time, the data is split by time rather than randomly,
  so models are always tested on payments that came after their training data.

## Missing values
- 214 of 434 columns are more than half empty; 12 are over 90% empty
  (mostly identity/device fields that only exist for some payments).
- Tree models handle missing values, so these columns are kept.

## Leakage check
- Each column was scored on its own (AUC). The best was `C4` at 0.68.
- No single column gives the answer away, so there is no obvious leak.
- Fraud only shows in combinations of signals, which tree ensembles are built to find.
- Excluded from model features: `TransactionID`, `TransactionDT`, `day`, `week`
  (they only record order and time).

## Splits
| Split  | Days    | Payments | Fraud rate |
|--------|---------|----------|------------|
| Train  | 1–120   | 414,542  | 3.52%      |
| Valid  | 121–152 | 90,568   | 3.39%      |
| Replay | 153–182 | 85,430   | 3.50%      |

Fraud rates are similar across splits, so later comparisons are fair.

## Setup note
Kaggle mounts competition data under `/kaggle/input/competitions/`, so the notebook
searches for the files instead of using a fixed path.