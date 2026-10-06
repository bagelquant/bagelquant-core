# `log_rank`

Apply a natural logarithm to same-date cross-sectional percentile ranks.

## Contract

Registry: `bagelquant_core.operator.logarithmic.log_rank`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
log_rank(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
log_rank(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | -1.09861 | 0 | -0.405465 |
| 2024-01-03 | missing | -0.693147 | 0 |
