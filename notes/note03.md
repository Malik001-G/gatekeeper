#  Models 2 and 3: bagging and random forest

Notebook: `notebooks/03_bagging_random_forest.ipynb`, trained on all 414,542 training rows
(4 CPU cores on Kaggle). Same features, splits and scoring as the decision tree.

## Bagging (50 trees)
- Each tree: grown deep (`min_samples_leaf = 5`) on a random 25% sample drawn with replacement.
- Trained in 404 s.

| Trees averaged | Valid PR-AUC |
|---|---|
| 1  | 0.256 |
| 2  | 0.360 |
| 5  | 0.431 |
| 10 | 0.463 |
| 20 | 0.493 |
| 50 | **0.504** |

- One deep tree alone (0.256) is worse than the pruned tree (0.368); two averaged already match it.
- Gains flatten after ~20 trees: averaging removes variance, with diminishing returns.

## Out-of-bag vs valid
- OOB PR-AUC (same months as training): **0.675**
- Valid PR-AUC (later months): **0.504**
- The 0.17 gap comes from fraud patterns shifting over time. OOB or a random split would
  overstate real performance by about a third, which confirms the time-based split.

## Random forest
`max_features` scan (100 trees each):

| max_features | Columns per split | Valid PR-AUC | Time |
|---|---|---|---|
| sqrt | ~20 | 0.485 | 28 s |
| 5%   | ~21 | 0.484 | 29 s |
| 15%  | ~64 | **0.502** | 90 s |

Final: 300 trees, `max_features = 15%`, 25% samples, `min_samples_leaf = 5`, trained in 267 s.

## Results so far
| Model | PR-AUC | Recall @ 1% false alarms | Cost | Train time | Time per payment |
|---|---|---|---|---|---|
| Decision tree (pruned) | 0.368 | 31.0% | $420,500 | 30 s | 3.6 ms |
| Bagging (50 trees) | **0.504** | **42.6%** | **$321,614** | 404 s | 7.8 ms |
| Random forest (300 trees) | **0.504** | 41.6% | $322,172 | 267 s | 17.0 ms |

- Both ensembles save ~$99k over the single tree and catch 11–12 points more fraud.
- Quality is tied. The forest trains faster but takes over twice as long per payment
  (300 trees voting vs 50). Both are within the 50 ms target.

## Features
- Random forest top 6: V258, C1, C13, V257, V201, card2.
- V258, C1 and C13 also led the single tree, so those signals are consistent, not chance.
- The forest draws on a broader set, including TransactionAmt, card1 and addr1.
