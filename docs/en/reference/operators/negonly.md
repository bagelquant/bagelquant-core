# `negonly`

Keep non-positive values, including zero, and mask positive values.

## Contract

Registry: `bagelquant_core.operator.temporal.negonly`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
negonly(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
negonly(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -2 | 0 | 1.2 |
| 2024-01-03 | -0.5 | 0.2 | 2.7 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -2 | 0 | missing |
| 2024-01-03 | -0.5 | missing | missing |
