# `cross_sectional_weighted_mean`

Broadcast each date's finite positive-weighted mean to its admitted asset coordinates.

## Contract

Registry: `bagelquant_core.operator.robust.cross_sectional_weighted_mean`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `sparse_ok`; traces: `custom`.

## Numerical signature

```python
cross_sectional_weighted_mean(frame: 'pl.DataFrame', *, weights: 'pl.DataFrame', positive_weights: 'bool' = True) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
cross_sectional_weighted_mean(source, weights=weights)
```

### source

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 6 | 10 |

### Auxiliary: weights

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 2 | 3 |

### Output

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 6.57143 | 6.57143 | 6.57143 | 6.57143 |
