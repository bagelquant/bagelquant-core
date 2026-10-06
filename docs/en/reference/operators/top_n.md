# `top_n`

Select exactly N finite scores, breaking ties by asset ID; incomplete cross-sections remain unavailable.

## Contract

Registry: `bagelquant_core.operator.portfolio.top_n`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `parent_max`.

## Numerical signature

```python
top_n(frame: 'pl.DataFrame', *, count: 'int' = 20) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
top_n(source, count=2)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0 | 1 | 1 |
| 2024-01-03 | 0 | 1 | 1 |
