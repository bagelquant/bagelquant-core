# `posonly`

Keep non-negative values, including zero, and mask negative values.

## Contract

Registry: `bagelquant_core.operator.temporal.posonly`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
posonly(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
posonly(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -2 | 0 | 1.2 |
| 2024-01-03 | -0.5 | 0.2 | 2.7 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | 0 | 1.2 |
| 2024-01-03 | missing | 0.2 | 2.7 |
