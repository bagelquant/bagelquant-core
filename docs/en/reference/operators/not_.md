# `not_`

Return one where elements are falsy and zero where they are truthy.

## Contract

Registry: `bagelquant_core.operator.comparison.not_`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `passthrough`.

## Numerical signature

```python
not_(frame: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
not_(source)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0 | missing | 0 |
| 2024-01-03 | 1 | 1 | 2 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | missing | 1 |
| 2024-01-03 | 0 | 0 | 0 |
