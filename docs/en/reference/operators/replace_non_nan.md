# `replace_non_nan`

Replace every non-missing value with the configured scalar.

## Contract

Registry: `bagelquant_core.operator.replace.replace_non_nan`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
replace_non_nan(frame: 'pl.DataFrame', *, value: 'Real') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
replace_non_nan(source, value=1.0)
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
| 2024-01-03 | missing | 1 | 1 |
