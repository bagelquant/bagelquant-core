# `coalesce`

Return the first non-missing value from the supplied inputs for each cell.

## Contract

Registry: `bagelquant_core.operator.combination.coalesce`. Version: `1`.

Peer inputs: 2 to N typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
coalesce(*frames: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
coalesce(input_1, input_2)
```

### input_1

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 3 |
| 2024-01-03 | missing | missing | inf |

### input_2

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 10 | 30 | 50 |
| 2024-01-03 | 20 | 40 | 60 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 30 | 3 |
| 2024-01-03 | 20 | 40 | inf |
