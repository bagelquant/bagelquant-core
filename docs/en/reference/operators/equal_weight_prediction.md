# `equal_weight_prediction`

Average every available policy-processed AlphaValue per asset.

## Contract

Registry: `prediction:equal_weight`. Version: `1`.

Peer inputs: 1 to N typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
equal_weight_prediction()
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
equal_weight_prediction(source, second)
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

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 2.5 |
| 2024-01-03 | 2 | 1.5 | 6 |
