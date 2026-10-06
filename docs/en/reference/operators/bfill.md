# `bfill`

Fill each asset's missing rows from later observations in time order, optionally bounded by `limit`.

## Contract

Registry: `bagelquant_core.operator.missing.bfill`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `backward_fill`.

## Numerical signature

```python
bfill(frame: 'pl.DataFrame', *, limit: 'int | None' = None) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
bfill(source, limit=1)
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
| 2024-01-03 | missing | missing | inf |
