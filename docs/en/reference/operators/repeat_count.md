# `repeat_count`

Count consecutive equal valid values within each asset in time order.

## Contract

Registry: `bagelquant_core.operator.streaks.repeat_count`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `custom`.

## Numerical signature

```python
repeat_count(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
repeat_count(source)
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
| 2024-01-02 | 1 | 1 |
| 2024-01-03 | 2 | 1 |
| 2024-01-04 | 1 | 1 |
| 2024-01-05 | 2 | 2 |
| 2024-01-08 | 1 | 1 |
