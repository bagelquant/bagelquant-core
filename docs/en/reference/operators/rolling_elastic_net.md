# `rolling_elastic_net`

Predict the current target from factors fitted only on prior per-asset rows with elastic-net regularization.

## Contract

Registry: `bagelquant_core.operator.regression.rolling_elastic_net`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `custom`.

## Numerical signature

```python
rolling_elastic_net(target: 'pl.DataFrame', *, factors: 'tuple[pl.DataFrame, ...]', window: 'int', alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, max_iter: 'int' = 1000, tolerance: 'float' = 1e-08) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_elastic_net(source, factors=(factors,), window=2)
```

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 2 | 3 |
| 2024-01-04 | 4 | 5 |
| 2024-01-05 | 7 | 8 |

### Auxiliary: factors

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
| 2024-01-04 | 1.5 | 2.5 |
| 2024-01-05 | 3 | 4 |
