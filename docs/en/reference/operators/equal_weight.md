# `equal_weight`

Convert a complete binary selection to equal positive target weights.

## Contract

Registry: `bagelquant_core.operator.portfolio.equal_weight`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `parent_max`.

## Numerical signature

```python
equal_weight(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
equal_weight(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 1 |
| 2024-01-03 | 0 | 0 | 0 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0.333333 | 0.333333 | 0.333333 |
| 2024-01-03 | missing | missing | missing |
