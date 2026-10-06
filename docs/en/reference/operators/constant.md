# `constant`

Replace every active-domain cell, including a missing value, with the configured constant.

## Contract

Registry: `bagelquant_core.operator.temporal.constant`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `passthrough`.

## Numerical signature

```python
constant(frame: 'pl.DataFrame', *, value: 'float' = 1) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
constant(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 1 |
| 2024-01-03 | 1 | 1 | 1 |
