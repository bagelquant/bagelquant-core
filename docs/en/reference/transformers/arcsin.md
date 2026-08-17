# `arcsin`

Return the inverse sine of each element, masking values outside `[-1, 1]`.

## Signature

```python
arcsin(source, *, name=None, metadata=None)
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
arcsin(source)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | -2.0 |
| 2024-01-02 | b | 0.0 |
| 2024-01-02 | c | 1.0 |
| 2024-01-03 | a | -1.0 |
| 2024-01-03 | b | 0.5 |
| 2024-01-03 | c | 2.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | missing |
| 2024-01-02 | b | 0.0 |
| 2024-01-02 | c | 1.5707963267948966 |
| 2024-01-03 | a | -1.5707963267948966 |
| 2024-01-03 | b | 0.5235987755982989 |
| 2024-01-03 | c | missing |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
