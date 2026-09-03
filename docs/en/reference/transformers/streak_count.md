# `streak_count`

Count consecutive increases or decreases within each asset in time order.

## Signature

```python
streak_count(source, *, reset_on_equal=True, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**reset_on_equal** : bool, default `True`
: Whether an unchanged value resets the directional streak to zero.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
streak_count(source)
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 3 |
| 2024-01-03 | 1 | 2 |
| 2024-01-04 | 2 | 1 |
| 2024-01-05 | 2 | 1 |
| 2024-01-08 | 3 | 0 |

### Output

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 0 | 0 |
| 2024-01-03 | 0 | -1 |
| 2024-01-04 | 1 | -2 |
| 2024-01-05 | 0 | 0 |
| 2024-01-08 | 1 | -1 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
