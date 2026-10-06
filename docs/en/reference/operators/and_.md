# `and_`

Return one where both corresponding elements are truthy and zero elsewhere.

## Contract

Registry: `bagelquant_core.operator.comparison.and_`. Version: `1`.

Peer inputs: 2 to 2 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
and_(lhs: 'pl.DataFrame', rhs: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
and_(input_1, input_2)
```

### input_1

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0 | missing | 0 |
| 2024-01-03 | 1 | 1 | 2 |

### input_2

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0 | 1 | 2 |
| 2024-01-03 | 0 | 1 | missing |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0 | missing | 0 |
| 2024-01-03 | 0 | 1 | missing |
