# `rolling_ridge`

Predict the current target from factors fitted only on prior per-asset rows with L2 regularization.

## Contract

Registry: `bagelquant_core.operator.regression.rolling_ridge`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `custom`.

## Numerical signature

```python
rolling_ridge(target: 'pl.DataFrame', *, factors: 'tuple[pl.DataFrame, ...]', window: 'int', alpha: 'float' = 1.0) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_ridge(source, factors=(factors,), window=2)
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
| 2024-01-04 | 1.66667 | 2.83333 |
| 2024-01-05 | 3.55556 | 4.77778 |
