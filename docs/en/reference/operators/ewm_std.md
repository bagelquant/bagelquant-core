# `ewm_std`

Return the per-asset exponentially weighted moving standard deviation in time order.

## Contract

Registry: `bagelquant_core.operator.rolling_stats.ewm_std`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `cumulative_max`.

## Numerical signature

```python
ewm_std(frame: 'pl.DataFrame', *, com: 'float | None' = None, span: 'float | None' = None, halflife: 'float | None' = None, alpha: 'float | None' = None, min_periods: 'int' = 0, adjust: 'bool' = True, ignore_na: 'bool' = False, bias: 'bool' = False) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
ewm_std(source, span=2.0)
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
| 2024-01-02 | 0 | 0 |
| 2024-01-03 | 0.707107 | 0.707107 |
| 2024-01-04 | 1.56893 | 1.56893 |
| 2024-01-05 | 2.62532 | 2.62532 |
