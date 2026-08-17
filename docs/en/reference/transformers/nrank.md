# `nrank`

Return a normalized same-date cross-sectional rank centered around zero.

## Signature

```python
nrank(source, *, name=None, metadata=None)
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
nrank(source)
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

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | -0.33333333333333337 |
| 2024-01-02 | b | 1.0 |
| 2024-01-02 | c | 0.33333333333333326 |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | 0.0 |
| 2024-01-03 | c | 1.0 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
