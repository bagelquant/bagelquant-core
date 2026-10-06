# `notnan`

Return one for present values and zero for null or NaN values.

## Contract

Registry: `bagelquant_core.operator.temporal.notnan`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
notnan(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
notnan(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | missing | missing | inf |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 0 | 1 |
| 2024-01-03 | 0 | 0 | 1 |
