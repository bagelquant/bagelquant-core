# `project`

Keep source values only where the key-aligned binary Node equals one.

## Contract

Registry: `bagelquant_core.operator.combination.project`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
project(frame: 'pl.DataFrame', *, binary: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
project(source, binary=binary)
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Auxiliary: binary

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 1 |
| 2024-01-03 | 0 | 0 | 0 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | missing | missing |
