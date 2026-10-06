# `streak_count`

Count consecutive increases or decreases within each asset in time order.

## Contract

Registry: `bagelquant_core.operator.streaks.streak_count`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `custom`.

## Numerical signature

```python
streak_count(frame: 'pl.DataFrame', *, reset_on_equal: 'bool' = True) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
streak_count(source)
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
| 2024-01-02 | 0 | 0 |
| 2024-01-03 | 0 | -1 |
| 2024-01-04 | 1 | -2 |
| 2024-01-05 | 0 | 0 |
| 2024-01-08 | 1 | -1 |
