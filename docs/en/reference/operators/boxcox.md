# `boxcox`

Apply the Box-Cox power transform element-wise with the supplied lambda.

## Contract

Registry: `bagelquant_core.operator.boxcox.boxcox`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
boxcox(frame: 'pl.DataFrame', *, lambda_: 'float' = 0) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
boxcox(source, lambda_=0.5)
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
| 2024-01-03 | missing | -0.585786 | 0.828427 |
