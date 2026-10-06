# `rebalance`

Select complete target snapshots on an explicit calendar anchored at Data Start.

## Contract

Registry: `bagelquant_core.operator.portfolio.rebalance`. Version: `1`.

Peer inputs: 1 to 1 typed Nodes. Returns a deferred Node.

Execution: `eager_barrier`; density: `dense_required`; traces: `parent_max`.

## Numerical signature

```python
rebalance(frame: 'pl.DataFrame', *, every: 'int' = 5, anchor: 'str' = 'data_start', calendar: 'pl.DataFrame', data_start: 'str') -> 'pl.DataFrame'
```

DataFrame arguments are Node dependencies in public calls. Scalar configuration is separate. Call `.compute()` to materialize.

## Executable example

```python
rebalance(source, calendar=calendar, every=1, data_start='2024-01-02')
```

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |

### Auxiliary: calendar

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2 | 2 |
| 2024-01-03 | 2 | 1 | 4 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | missing | missing |
