# `mad_zscore`

Clip each finite cross-section by median/MAD and standardize with population variance.

## Contract

Registry: `bagelquant_core.operator.robust.mad_zscore`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `sparse_ok`; traces: `custom`.

## Numerical signature

```python
mad_zscore(frame: 'pl.DataFrame', *, clip_multiplier: 'float' = 5.0, mad_scale: 'float' = 1.4826, min_scale: 'float' = 1e-12) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
mad_zscore(source)
```

### source

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 6 | 10 |

### Output

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | -1.17954 | -0.589768 | 0.294884 | 1.47442 |
