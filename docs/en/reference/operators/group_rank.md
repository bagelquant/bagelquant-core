# `group_rank`

Return average-tie ordinal ranks within each declared group and date.

## Contract

Registry: `bagelquant_core.operator.cross_sectional.group_rank`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `lazy`; density: `sparse_ok`; traces: `parent_max`.

## Numerical signature

```python
group_rank(frame: 'pl.DataFrame', *, group: 'pl.DataFrame') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
group_rank(source, group=group)
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
| 2024-01-02 | 1 | 2 | 1 | 2 |
