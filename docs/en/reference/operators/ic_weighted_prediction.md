# `ic_weighted_prediction`

Combine AlphaValues with positive rolling mean Spearman IC.

## Contract

Registry: `prediction:ic_weighted`. Version: `1`.

Peer inputs: 1 to N typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
ic_weighted_prediction(window: 'int') -> 'None'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
ic_weighted_prediction(source, second, targets=targets, availability=availability, window=2)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### second

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2 | 2 |
| 2024-01-03 | 2 | 1 | 4 |

### Auxiliary: targets

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2 | 2 |
| 2024-01-03 | 2 | 1 | 4 |

### Auxiliary: availability

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 738886 | 738886 | 738886 |
| 2024-01-03 | 738886 | 738886 | 738886 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | missing | missing | missing |
| 2024-01-03 | missing | missing | missing |
