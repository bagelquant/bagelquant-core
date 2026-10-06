# `arctanh`

Return the inverse hyperbolic tangent of each element, masking values outside `(-1, 1)`.

## Contract

Registry: `bagelquant_core.operator.trigonometric.arctanh`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
arctanh(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
arctanh(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -2 | 0 | 1 |
| 2024-01-03 | -1 | 0.5 | 2 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | 0 | missing |
| 2024-01-03 | missing | 0.549306 | missing |
