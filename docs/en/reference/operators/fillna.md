# `fillna`

Replace missing panel values with the configured scalar.

## Contract

Registry: `bagelquant_core.operator.missing.fillna`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `passthrough`.

## Numerical signature

```python
fillna(frame: 'pl.DataFrame', *, value: 'float' = 0) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
fillna(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | 0 | 2 | 8 |
