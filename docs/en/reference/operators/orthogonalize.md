# `orthogonalize`

Return same-date cross-sectional residuals after regressing the source on the named factor Panels.

## Contract

Registry: `bagelquant_core.operator.cross_sectional.orthogonalize`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
orthogonalize(frame: 'pl.DataFrame', *, factors: 'tuple[pl.DataFrame, ...]', fit_intercept: 'bool' = False) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
orthogonalize(source, factors=(factors,))
```

### source

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 6 | 10 |

### Auxiliary: factors

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 2 | 3 |

### Output

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | -2.06667 | -0.0666667 | -0.133333 | 0.8 |
