# `rolling_corr`

Return the trailing per-asset correlation between two aligned panels.

## Contract

Registry: `bagelquant_core.operator.regression.rolling_corr`. Version: `1`.

Peer inputs: 2 to 2 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `custom`.

## Numerical signature

```python
rolling_corr(lhs: 'pl.DataFrame', rhs: 'pl.DataFrame', *, window: 'int', min_periods: 'int | None' = None) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rolling_corr(input_1, input_2, window=2)
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
| 2024-01-03 | 1 | 1 |
| 2024-01-04 | 1 | 1 |
| 2024-01-05 | 1 | 1 |
