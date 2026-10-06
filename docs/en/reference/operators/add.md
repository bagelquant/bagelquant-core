# `add`

Add two key-aligned panel values element-wise.

## Contract

Registry: `bagelquant_core.operator.arithmetic.add`. Version: `1`.

Peer inputs: 2 to 2 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
add(lhs: 'pl.DataFrame', rhs: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
add(input_1, input_2)
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
| 2024-01-02 | 2 | 6 | 5 |
| 2024-01-03 | missing | 3 | 12 |
