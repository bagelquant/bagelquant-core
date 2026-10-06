# `regularized_weights`

Optimize capped long-only targets using preceding target weights as causal state.

## Contract

Registry: `bagelquant_core.operator.portfolio.regularized_weights`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `parent_max`.

## Numerical signature

```python
regularized_weights(frame: 'pl.DataFrame', *, concentration_penalty: 'float' = 10.0, turnover_penalty: 'float' = 0.1, max_weight: 'float' = 0.04, tolerance: 'float' = 1e-07, reference: 'pl.DataFrame | None' = None) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
regularized_weights(source, max_weight=0.5)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0.25 | 0.4 | 0.35 |
| 2024-01-03 | 0 | 0.5 | 0.5 |
