# `power`

Raise each present source value to the configured scalar exponent.

## Contract

Registry: `bagelquant_core.operator.power.power`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
power(frame: 'pl.DataFrame', *, exponent: 'Real') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
power(source, exponent=2.0)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 16 | 9 |
| 2024-01-03 | missing | 4 | 64 |
