# `group_std`

Replace each value with its declared group's same-date standard deviation.

## Contract

Registry: `bagelquant_core.operator.cross_sectional.group_std`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
group_std(frame: 'pl.DataFrame', *, group: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
group_std(source, group=group)
```

### source

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 6 | 10 |

### Auxiliary: group

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | tech | tech | bank | bank |

### Output

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1.41421 | 1.41421 | 2.82843 | 2.82843 |
