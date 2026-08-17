# `lag`

Return the value `periods` earlier within each asset ordered by time.

## Signature

```python
lag(source, *, periods=1, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**periods** : int, default `1`
: Number of prior rows to shift or compare. Must be a positive integer.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
lag(source)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

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

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | missing |
| 2024-01-02 | b | missing |
| 2024-01-03 | a | 1.0 |
| 2024-01-03 | b | 2.0 |
| 2024-01-04 | a | 2.0 |
| 2024-01-04 | b | 3.0 |
| 2024-01-05 | a | 4.0 |
| 2024-01-05 | b | 5.0 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
