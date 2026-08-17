# `nonnans`

Replace null and NaN values with zero.

## Signature

```python
nonnans(source, *, name=None, metadata=None)
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
nonnans(source)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | missing |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | missing |
| 2024-01-03 | c | inf |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 0.0 |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | 0.0 |
| 2024-01-03 | b | 0.0 |
| 2024-01-03 | c | inf |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
