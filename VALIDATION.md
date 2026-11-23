# Validation

The repaired train.py entry point passed fresh execution. The original exploratory notebooks are retained and are not all reproduced end to end.

Measured local baseline output:

```json
{
  "rows": 68462,
  "split": "random 80/20, seed 42",
  "regression_model": "Ridge alpha=10",
  "mae": 2332.3266590957746,
  "rmse": 3599.4346331339075,
  "r2": 0.8494081304054036,
  "knn_tier_accuracy": 0.8795004746950997,
  "train_only_price_cutoffs": [
    11490.0,
    17999.0
  ]
}

```

These are a single seeded holdout result, not future-performance guarantees.
