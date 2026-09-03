# `smooth`

Smooth each per-asset time series over 10 sessions with a 3-session half-life and at least 5 valid observations.

## Signature

```python
smooth(source, *, name=None, metadata=None)
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
smooth(source)
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 2 | 4 |
| 2024-01-04 | 3 | missing |
| 2024-01-05 | 4 | 8 |
| 2024-01-08 | 5 | 10 |
| 2024-01-09 | 6 | 12 |
| 2024-01-10 | 7 | 14 |
| 2024-01-11 | 8 | 16 |
| 2024-01-12 | 9 | 18 |
| 2024-01-15 | 10 | 20 |
| 2024-01-16 | 11 | 22 |

### Output

| time | a | b |
|---|---:|---:|
| 2024-01-02 | missing | missing |
| 2024-01-03 | missing | missing |
| 2024-01-04 | missing | missing |
| 2024-01-05 | missing | missing |
| 2024-01-08 | 3.45174 | missing |
| 2024-01-09 | 4.15268 | 8.67298 |
| 2024-01-10 | 4.88549 | 10.1999 |
| 2024-01-11 | 5.64812 | 11.7389 |
| 2024-01-12 | 6.43839 | 13.3075 |
| 2024-01-15 | 7.25408 | 14.9132 |
| 2024-01-16 | 8.25408 | 16.9013 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
