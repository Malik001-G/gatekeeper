# Models 4 and 5: AdaBoost and gradient boosting (LightGBM)

Notebook: `notebooks/04_boosting.ipynb`, trained on all 414,542 training rows (4 CPU cores on Kaggle).
Same features, splits and scoring as the earlier models.

## AdaBoost
Tree size chosen on a 120k-row sample (50 rounds each):

| Base tree | Valid PR-AUC | Time |
|---|---|---|
| depth 1 | 0.334 | 40 s |
| depth 2 | 0.370 | 75 s |
| depth 3 | **0.409** | 109 s |

Final: depth-3 trees, 100 rounds, learning rate 0.5, all training rows, 893 s.

| Rounds | Valid PR-AUC |
|---|---|
| 1   | 0.161 |
| 10  | 0.337 |
| 25  | 0.377 |
| 50  | 0.391 |
| 100 | **0.401** |

- Plateaus around 0.40, only slightly above the pruned tree.
- AdaBoost raises the weight of misjudged payments each round; with rare, noisy fraud labels
  it spends effort on unusual or mislabelled cases.
- Each round depends on the last, so it runs on one core: slowest to train and to score.

## LightGBM
Settings: learning rate 0.05, up to 127 leaves per tree, 80% row and 50% column sampling,
early stopping after 100 rounds without improvement.

- Passes bagging and random forest after ~35 trees.
- Best at **472 trees** (stopped at 572), trained in 170 s.
- Early stopping used `valid`, so the valid score is slightly favourable; the untouched
  `replay` split is the final test.

## Full contest
| Model | PR-AUC | Recall @ 1% false alarms | Cost | Train time | Time per payment |
|---|---|---|---|---|---|
| **LightGBM** | **0.620** | **52.8%** | **$256,208** | 170 s | **2.4 ms** |
| Random forest (300 trees) | 0.504 | 41.6% | $322,172 | 267 s | 17.0 ms |
| Bagging (50 trees) | 0.504 | 42.6% | $321,614 | 404 s | 7.8 ms |
| AdaBoost (100 rounds) | 0.401 | 32.7% | $391,691 | 893 s | 22.8 ms |
| Decision tree (pruned) | 0.368 | 31.0% | $420,500 | 30 s | 3.6 ms |
| Never flag (baseline) | — | 0% | $492,574 | — | — |

- LightGBM leads on every measure: ~$66k less lost than the forest and ~$236k (48%) less
  than flagging nothing, while scoring a payment ~7x faster than the forest.

## Features (LightGBM, share of gain)
- Top 6: V258, C13, C1, card1, card2, TransactionAmt.
- V258 leads in every model.
- LightGBM also draws on addr1, P_emaildomain and the D columns (time gaps between events),
  which earlier models used little.

## Saved for the API
- `lightgbm_model.txt` (6.6 MB): the trained model at its best round.
- `preprocess.json`: feature order and text codes, so live payments are prepared identically.
