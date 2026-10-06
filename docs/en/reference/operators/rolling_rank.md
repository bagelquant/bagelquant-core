# `rolling_rank`

Return the trailing window's latest valid value as an average-tie rank among valid observations.

## Contract

Registry: `bagelquant_core.operator.rolling_stats.rolling_rank`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `rolling_max`.

## Numerical signature

```python
rolling_rank(frame: 'pl.DataFrame', *, window: 'int', min_periods: 'int | None' = None) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_rank(source, window=2)
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
| 2024-01-03 | 2 | 2 |
| 2024-01-04 | 2 | 2 |
| 2024-01-05 | 2 | 2 |
