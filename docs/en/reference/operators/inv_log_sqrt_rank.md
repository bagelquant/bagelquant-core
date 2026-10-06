# `inv_log_sqrt_rank`

Return `-log(rank) / sqrt(rank)` using cross-sectional percentile ranks.

## Contract

Registry: `bagelquant_core.operator.logarithmic.inv_log_sqrt_rank`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
inv_log_sqrt_rank(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
inv_log_sqrt_rank(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1.90285 | -0 | 0.496591 |
| 2024-01-03 | missing | 0.980258 | -0 |
