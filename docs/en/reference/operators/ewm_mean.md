# `ewm_mean`

Return the per-asset exponentially weighted moving mean in time order.

## Contract

Registry: `bagelquant_core.operator.rolling_stats.ewm_mean`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `cumulative_max`.

## Numerical signature

```python
ewm_mean(frame: 'pl.DataFrame', *, com: 'float | None' = None, span: 'float | None' = None, halflife: 'float | None' = None, alpha: 'float | None' = None, min_periods: 'int' = 0, adjust: 'bool' = True, ignore_na: 'bool' = False) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
ewm_mean(source, span=2.0)
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
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 1.75 | 2.75 |
| 2024-01-04 | 3.30769 | 4.30769 |
| 2024-01-05 | 5.8 | 6.8 |
