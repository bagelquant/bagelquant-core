# `ffill`

Fill each asset's missing rows from earlier observations in time order, optionally bounded by `limit`.

## Contract

Registry: `bagelquant_core.operator.missing.ffill`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `forward_fill`.

## Numerical signature

```python
ffill(frame: 'pl.DataFrame', *, limit: 'int | None' = None) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
ffill(source, limit=1)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | missing | missing | inf |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | 1 | missing | inf |
