# `repeat_count`

Count consecutive equal valid values within each asset in time order.

## Signature

```python
repeat_count(source, *, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
repeat_count(source)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 3.0 |
| 2024-01-03 | a | 1.0 |
| 2024-01-03 | b | 2.0 |
| 2024-01-04 | a | 2.0 |
| 2024-01-04 | b | 1.0 |
| 2024-01-05 | a | 2.0 |
| 2024-01-05 | b | 1.0 |
| 2024-01-08 | a | 3.0 |
| 2024-01-08 | b | 0.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1 |
| 2024-01-02 | b | 1 |
| 2024-01-03 | a | 2 |
| 2024-01-03 | b | 1 |
| 2024-01-04 | a | 1 |
| 2024-01-04 | b | 1 |
| 2024-01-05 | a | 2 |
| 2024-01-05 | b | 2 |
| 2024-01-08 | a | 1 |
| 2024-01-08 | b | 1 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
