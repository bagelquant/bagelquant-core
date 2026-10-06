# `log`

Return the natural logarithm of positive values and mask invalid inputs.

## Contract

Registry: `bagelquant_core.operator.logarithmic.log`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
log(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
log(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -2 | 0 | 1 |
| 2024-01-03 | -1 | 0.5 | 2 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | missing | 0 |
| 2024-01-03 | missing | -0.693147 | 0.693147 |
