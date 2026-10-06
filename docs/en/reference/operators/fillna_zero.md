# `fillna_zero`

Replace null and NaN panel values with zero.

## Contract

Registry: `bagelquant_core.operator.missing.fillna_zero`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `passthrough`.

## Numerical signature

```python
fillna_zero(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
fillna_zero(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | missing | missing | inf |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 0 | 3 |
| 2024-01-03 | 0 | 0 | inf |
