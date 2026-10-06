# `rolling_ewm_fw`

Return a finite-window exponentially weighted per-asset mean in time order.

## Contract

Registry: `bagelquant_core.operator.rolling_stats.rolling_ewm_fw`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `rolling_max`.

## Numerical signature

```python
rolling_ewm_fw(frame: 'pl.DataFrame', *, window: 'int', halflife: 'float', min_periods: 'int' = 0) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_ewm_fw(source, window=2, halflife=2.0)
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
| 2024-01-03 | 1.58579 | 2.58579 |
| 2024-01-04 | 3.17157 | 4.17157 |
| 2024-01-05 | 5.75736 | 6.75736 |
