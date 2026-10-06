# `rolling_lightgbm_prediction`

Fit deterministic CPU LightGBM using explicit matured labels and resource budgets.

## Contract

Registry: `bagelquant_core.operator.training.rolling_lightgbm_prediction`. Version: `2`.

Peer inputs: 1 to N typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `parent_max`.

## Numerical signature

```python
rolling_lightgbm_prediction(*features: 'pl.DataFrame', labels: 'pl.DataFrame', label_end: 'pl.DataFrame | None' = None, label_available: 'pl.DataFrame | None' = None, window: 'int' = 252, fit_every: 'int' = 20, min_samples: 'int' = 200, max_samples: 'int' = 100000, label_maturity: 'int' = 2, anchor_offset: 'int' = 0, num_leaves: 'int' = 31, learning_rate: 'float' = 0.05, min_data_in_leaf: 'int' = 20, lambda_l1: 'float' = 0.0, lambda_l2: 'float' = 1.0, feature_fraction: 'float' = 1.0, num_boost_round: 'int' = 100, max_bin: 'int' = 255, max_depth: 'int' = -1, min_gain_to_split: 'float' = 0.0, seed: 'int' = 1729) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_lightgbm_prediction(source, labels=labels, window=3, fit_every=1, min_samples=100, max_samples=100, label_maturity=1, num_boost_round=2, min_data_in_leaf=1)
```

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 2 | 3 |
| 2024-01-04 | 4 | 5 |
| 2024-01-05 | 7 | 8 |

### Auxiliary: labels

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 1 |
| 2024-01-03 | 1.5 | 2 |
| 2024-01-04 | 2 | 2.5 |
| 2024-01-05 | 3 | 4 |

### Output

| time | a | b |
|---|---:|---:|
| 2024-01-02 | missing | missing |
| 2024-01-03 | missing | missing |
| 2024-01-04 | missing | missing |
| 2024-01-05 | missing | missing |
