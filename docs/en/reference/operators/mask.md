# `mask`

Keep source values where the key-aligned mask is truthy and use the configured replacement elsewhere.

## Contract

Registry: `bagelquant_core.operator.combination.mask`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
mask(frame: 'pl.DataFrame', *, mask_frame: 'pl.DataFrame', replace_value: 'float' = nan) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
mask(source, mask_frame=mask_frame, replace_value=0.0)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Auxiliary: mask_frame

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 1 |
| 2024-01-03 | 0 | 0 | 0 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | 0 | 0 | 0 |
