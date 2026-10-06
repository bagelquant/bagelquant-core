# `pct_change`

Return each asset's percentage change from the observation `periods` rows earlier.

## Contract

Registry: `bagelquant_core.operator.basic.pct_change`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `current_and_shift_max`.

## Numerical signature

```python
pct_change(frame: 'pl.DataFrame', *, periods: 'int' = 1) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
pct_change(source)
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
| 2024-01-03 | 1 | 0.5 |
| 2024-01-04 | 1 | 0.666667 |
| 2024-01-05 | 0.75 | 0.6 |
