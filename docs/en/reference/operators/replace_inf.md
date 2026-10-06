# `replace_inf`

Replace positive and negative infinity with missing values.

## Contract

Registry: `bagelquant_core.operator.temporal.replace_inf`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
replace_inf(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
replace_inf(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | missing | missing | inf |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | missing | missing | missing |
