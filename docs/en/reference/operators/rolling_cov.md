# `rolling_cov`

Return the trailing per-asset covariance between two aligned panels.

## Contract

Registry: `bagelquant_core.operator.regression.rolling_cov`. Version: `1`.

Peer inputs: 2 to 2 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `custom`.

## Numerical signature

```python
rolling_cov(lhs: 'pl.DataFrame', rhs: 'pl.DataFrame', *, window: 'int', min_periods: 'int | None' = None, ddof: 'int' = 1) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_cov(input_1, input_2, window=2)
```

### input_1

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 2 | 3 |
| 2024-01-04 | 4 | 5 |
| 2024-01-05 | 7 | 8 |

### input_2

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
| 2024-01-03 | 0.25 | 0.5 |
| 2024-01-04 | 0.5 | 0.5 |
| 2024-01-05 | 1.5 | 2.25 |
