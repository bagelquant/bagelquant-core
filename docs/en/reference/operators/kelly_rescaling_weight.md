# `kelly_rescaling_weight`

Calculate the rolling Kelly mean-to-variance ratio and clip it to `[0, 1]`.

## Contract

Registry: `bagelquant_core.operator.kelly.kelly_rescaling_weight`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `rolling_max`.

## Numerical signature

```python
kelly_rescaling_weight(frame: 'pl.DataFrame', *, window: 'int') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
kelly_rescaling_weight(source, window=2)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 2 | 3 | 1 |
| 2024-01-03 | missing | 2 | 4 |
| 2024-01-04 | 5 | 1 | 3 |
| 2024-01-05 | 2 | 6 | 1 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | missing | missing |
| 2024-01-03 | missing | 1 | 0.555556 |
| 2024-01-04 | missing | 1 | 1 |
| 2024-01-05 | 0.777778 | 0.28 | 1 |
