# `signed_log1p`

Return `sign(value) * log(1 + abs(value))` element-wise.

## Contract

Registry: `bagelquant_core.operator.logarithmic.signed_log1p`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
signed_log1p(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
signed_log1p(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0.693147 | 1.60944 | 1.38629 |
| 2024-01-03 | missing | 1.09861 | 2.19722 |
