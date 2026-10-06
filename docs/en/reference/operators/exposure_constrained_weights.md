# `exposure_constrained_weights`

Optimize target weights within explicit exposure bounds using historical target state.

## Contract

Registry: `bagelquant_core.operator.portfolio.exposure_constrained_weights`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `parent_max`.

## Numerical signature

```python
exposure_constrained_weights(frame: 'pl.DataFrame', *, exposures: 'tuple[pl.DataFrame, ...]', lower_bounds: 'list[float]', upper_bounds: 'list[float]', concentration_penalty: 'float' = 10.0, turnover_penalty: 'float' = 0.1, max_weight: 'float' = 0.04, max_turnover: 'float | None' = None, constraint_tolerance: 'float' = 1e-07, reference: 'pl.DataFrame | None' = None) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
exposure_constrained_weights(source, exposures=(exposures,), max_weight=1.0, lower_bounds=[-100.0], upper_bounds=[100.0])
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Auxiliary: exposures

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2 | 2 |
| 2024-01-03 | 2 | 1 | 4 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0.25 | 0.4 | 0.35 |
| 2024-01-03 | 0 | 0.355 | 0.645 |
