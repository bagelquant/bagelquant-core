# `denoise`

Set each value whose absolute magnitude is below the configured threshold to zero.

## Contract

Registry: `bagelquant_core.operator.temporal.denoise`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
denoise(frame: 'pl.DataFrame', *, threshold: 'float' = 1e-12) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
denoise(source, threshold=2.5)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0 | 4 | 3 |
| 2024-01-03 | missing | 0 | 8 |
