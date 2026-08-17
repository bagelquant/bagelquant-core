# `mask`

Keep source values where the key-aligned mask is truthy and use the configured replacement elsewhere.

## Signature

```python
mask(source, *, mask_frame, replace_value=nan, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**mask_frame** : Panel | Graph
: Mask input. Truthy cells retain values; false or missing cells are replaced.
**replace_value** : float, default `nan`
: Value inserted where the mask is false or missing.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
mask(source, mask_frame=mask_frame, replace_value=0.0)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 4.0 |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | 2.0 |
| 2024-01-03 | c | 8.0 |
### Panel parameter: mask_frame

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 1.0 |
| 2024-01-02 | c | 1.0 |
| 2024-01-03 | a | 0.0 |
| 2024-01-03 | b | 0.0 |
| 2024-01-03 | c | 0.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 4.0 |
| 2024-01-02 | c | 3.0 |
| 2024-01-03 | a | 0.0 |
| 2024-01-03 | b | 0.0 |
| 2024-01-03 | c | 0.0 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
