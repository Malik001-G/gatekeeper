# Day 2 — Model 1: decision tree

Notebook: `notebooks/02_decision_tree.ipynb`, using the train/valid splits from Day 1.
431 features (31 text columns encoded as numbers, missing values set to -999).

## Baseline to beat
A model that never flags fraud misses all 3,069 frauds in `valid`: **$492,574 lost**.

## Unlimited tree: overfitting
- Grew to depth 71 with 3,629 leaves (trained on a 120k-row sample in 25 s).
- PR-AUC on the rows it learned from: **1.000**. On unseen payments: **0.121**.
- It memorised the training data instead of learning general patterns.
- It only outputs 0% or 100% fraud, so it cannot be set to a 1% false-alarm rate;
  at that setting it catches no fraud.

## Pruning
- Depths 4–16 tried. Valid PR-AUC rose until **depth 10 (0.374)**, then flattened.
- Deeper trees kept improving on seen data but not on unseen data.
- Cost-complexity pruning: `ccp_alpha = 1e-05` cut leaves from 90 to 76 with no loss
  in score; `1e-04` cut too much (0.353).

## Final pruned tree
Depth 10, `ccp_alpha = 1e-05`, 114 leaves, trained on all 414,542 rows in 30 s.

| Metric | Value |
|---|---|
| PR-AUC | **0.368** (random guessing ≈ 0.035) |
| Fraud caught at 1% false alarms | **31%** |
| Cost | **$420,500** (~$72k saved vs never flagging) |
| Time per payment | 3.6 ms |

A clear improvement, but 69% of fraud still gets through.

## What the tree looks at
- First split: `V258 <= 2.5`. The 1.1% of payments above it are **61% fraud**
  (vs 3.5% overall); adding `V205 > 63.7` raises that to **85%**.
- Top features: V258, C1, C14, C7, V149, C13.
- V and C columns are anonymised (Vesta-engineered features and counts), so their
  importance is visible but their exact meaning is not.

## Training time
Each tree trained in ~25–30 s, so tuning on the full training set is practical for later models.
