# `rolling_corr`

Return the trailing per-asset correlation between two aligned panels.

## Signature

```python
rolling_corr(lhs, rhs, *, window, min_periods=None, name=None, metadata=None)
```

## Parameters

**lhs** : Panel | Graph
: Left-hand numeric `Panel` or single-output `Graph`.
**rhs** : Panel | Graph
: Right-hand numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
**min_periods** : int | None, default `None`
: Minimum number of observations required to produce a value.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
rolling_corr(input_1, input_2, window=2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### input_1

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 2.0 |
| 2024-01-03 | a | 2.0 |
| 2024-01-03 | b | 3.0 |
| 2024-01-04 | a | 4.0 |
| 2024-01-04 | b | 5.0 |
| 2024-01-05 | a | 7.0 |
| 2024-01-05 | b | 8.0 |
### input_2

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 1.0 |
| 2024-01-03 | a | 1.5 |
| 2024-01-03 | b | 2.0 |
| 2024-01-04 | a | 2.0 |
| 2024-01-04 | b | 2.5 |
| 2024-01-05 | a | 3.0 |
| 2024-01-05 | b | 4.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | missing |
| 2024-01-02 | b | missing |
| 2024-01-03 | a | 1.0 |
| 2024-01-03 | b | 1.0 |
| 2024-01-04 | a | 1.0 |
| 2024-01-04 | b | 1.0 |
| 2024-01-05 | a | 1.0 |
| 2024-01-05 | b | 1.0 |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
