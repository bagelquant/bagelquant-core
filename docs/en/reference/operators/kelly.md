# `kelly`

Estimate a rolling Kelly-style signal from each asset's trailing mean and variance.

## Contract

Registry: `bagelquant_core.operator.kelly.kelly`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `rolling_max`.

## Numerical signature

```python
kelly(frame: 'pl.DataFrame', *, window: 'int') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
kelly(source, window=2)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 2 | 3 | 1 |
| 2024-01-03 | missing | 2 | 4 |
| 2024-01-04 | 5 | 1 | 3 |
| 2024-01-05 | 2 | 6 | 1 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | missing | missing |
| 2024-01-03 | missing | 5 | 0.555556 |
| 2024-01-04 | missing | 3 | 7 |
| 2024-01-05 | 0.777778 | 0.28 | 1 |
