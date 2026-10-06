# `kelly_rank_boxcox`

Percentile-rank each date cross-section, apply Box-Cox, then calculate the rolling Kelly mean-to-variance ratio.

## Contract

Registry: `bagelquant_core.operator.kelly.kelly_rank_boxcox`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `rolling_max`.

## Numerical signature

```python
kelly_rank_boxcox(frame: 'pl.DataFrame', *, window: 'int', lambda_: 'float' = 0) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
kelly_rank_boxcox(source, window=2)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 2 | 3 | 1 |
| 2024-01-03 | missing | 2 | 4 |
| 2024-01-04 | 5 | 1 | 3 |
| 2024-01-05 | 2 | 6 | 1 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | missing | missing |
| 2024-01-03 | missing | -1.4427 | -0.910239 |
| 2024-01-04 | missing | -10.8987 | -2.4663 |
| 2024-01-05 | -2.4663 | -0.910239 | -3.13054 |
