# `arcsin`

Return the inverse sine of each element, masking values outside `[-1, 1]`.

## Contract

Registry: `bagelquant_core.operator.trigonometric.arcsin`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
arcsin(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
arcsin(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -2 | 0 | 1 |
| 2024-01-03 | -1 | 0.5 | 2 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | 0 | 1.5708 |
| 2024-01-03 | -1.5708 | 0.523599 | missing |
