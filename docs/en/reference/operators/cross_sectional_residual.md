# `cross_sectional_residual`

Return full-rank cross-sectional OLS residuals with explicit controls and intercept.

## Contract

Registry: `bagelquant_core.operator.robust.cross_sectional_residual`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `sparse_ok`; traces: `custom`.

## Numerical signature

```python
cross_sectional_residual(frame: 'pl.DataFrame', *, controls: 'tuple[pl.DataFrame, ...]', fit_intercept: 'bool' = True) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
cross_sectional_residual(source, controls=(controls,))
```

### source

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 6 | 10 |

### Auxiliary: controls

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 2 | 3 |

### Output

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | -1 | 1 | 1.77636e-15 | 3.55271e-15 |
