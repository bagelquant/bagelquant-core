# `lag`

Return the value `periods` earlier within each asset ordered by time.

## Contract

Registry: `bagelquant_core.operator.temporal.lag`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `dense_required`; traces: `shift`.

## Numerical signature

```python
lag(frame: 'pl.DataFrame', *, periods: 'int' = 1) -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
lag(source)
```

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 2 | 3 |
| 2024-01-04 | 4 | 5 |
| 2024-01-05 | 7 | 8 |

### Output

| time | a | b |
|---|---:|---:|
| 2024-01-02 | missing | missing |
| 2024-01-03 | 1 | 2 |
| 2024-01-04 | 2 | 3 |
| 2024-01-05 | 4 | 5 |
