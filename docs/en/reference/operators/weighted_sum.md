# `weighted_sum`

Return the element-wise weighted sum of aligned value and weight panels.

## Contract

Registry: `bagelquant_core.operator.aggregation.weighted_sum`. Version: `1`.

Peer inputs: 2 to N typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
weighted_sum(*frames: 'pl.DataFrame', weights: 'Sequence[float]') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
weighted_sum(input_1, input_2, weights=[0.25, 0.75])
```

### input_1

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### input_2

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2 | 2 |
| 2024-01-03 | 2 | 1 | 4 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2.5 | 2.25 |
| 2024-01-03 | missing | 1.25 | 5 |
