# `pct_change_from_last_change`

Return the adjacent percentage change only when the per-asset value changes.

## Contract

Registry: `bagelquant_core.operator.streaks.pct_change_from_last_change`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `current_and_shift_max`.

## Numerical signature

```python
pct_change_from_last_change(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
pct_change_from_last_change(source)
```

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 3 |
| 2024-01-03 | 1 | 2 |
| 2024-01-04 | 2 | 1 |
| 2024-01-05 | 2 | 1 |
| 2024-01-08 | 3 | 0 |

### Output

| time | a | b |
|---|---:|---:|
| 2024-01-02 | missing | missing |
| 2024-01-03 | missing | -0.333333 |
| 2024-01-04 | 1 | -0.5 |
| 2024-01-05 | missing | missing |
| 2024-01-08 | 0.5 | -1 |
