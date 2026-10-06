# `rolling_elastic_net_prediction`

Fit and predict RMS-scaled Elastic Net using only matured historical labels.

## Contract

Registry: `bagelquant_core.operator.training.rolling_elastic_net_prediction`. Version: `2`.

Peer inputs: 1 to N typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `parent_max`.

## Numerical signature

```python
rolling_elastic_net_prediction(*features: 'pl.DataFrame', labels: 'pl.DataFrame', label_end: 'pl.DataFrame | None' = None, label_available: 'pl.DataFrame | None' = None, window: 'int' = 252, fit_every: 'int' = 20, min_samples: 'int' = 200, max_samples: 'int' = 100000, label_maturity: 'int' = 2, anchor_offset: 'int' = 0, alpha: 'float' = 0.001, l1_ratio: 'float' = 0.5, max_iter: 'int' = 10000, tolerance: 'float' = 1e-06, seed: 'int' = 1729) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_elastic_net_prediction(source, labels=labels, window=3, fit_every=1, min_samples=2, max_samples=100, label_maturity=1)
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
| 2024-01-03 | 1 | 1 |
| 2024-01-04 | 2.36629 | 2.86194 |
| 2024-01-05 | 3.26127 | 3.64397 |
