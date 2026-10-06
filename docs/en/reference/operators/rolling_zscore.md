# `rolling_zscore`

Standardize the trailing window's latest valid value by that window's valid mean and standard deviation.

## Contract

Registry: `bagelquant_core.operator.rolling_stats.rolling_zscore`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `rolling_max`.

## Numerical signature

```python
rolling_zscore(frame: 'pl.DataFrame', *, window: 'int', min_periods: 'int | None' = None, ddof: 'int' = 1) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_zscore(source, window=2)
```

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 2 | 3 |
| 2024-01-04 | 4 | 5 |
| 2024-01-05 | 7 | 8 |

### Output

| time | a | b |
|---|---:|---:|
| 2024-01-02 | missing | missing |
| 2024-01-03 | 0.707107 | 0.707107 |
| 2024-01-04 | 0.707107 | 0.707107 |
| 2024-01-05 | 0.707107 | 0.707107 |
