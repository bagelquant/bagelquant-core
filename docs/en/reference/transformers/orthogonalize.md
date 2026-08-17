# `orthogonalize`

Return same-date cross-sectional residuals after regressing the source on the named factor Panels.

## Signature

```python
orthogonalize(source, *, factors, fit_intercept=False, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**factors** : tuple[pl.DataFrame, ...]
: One or more factor `Panel` or single-output `Graph` inputs.
**fit_intercept** : bool, default `False`
: Whether the cross-sectional or rolling regression includes an intercept.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
orthogonalize(source, factors=(factors,))
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 3.0 |
| 2024-01-02 | c | 6.0 |
| 2024-01-02 | d | 10.0 |
### Panel parameter: factors

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 1.0 |
| 2024-01-02 | c | 2.0 |
| 2024-01-02 | d | 3.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | -2.066666666666667 |
| 2024-01-02 | b | -0.06666666666666687 |
| 2024-01-02 | c | -0.13333333333333375 |
| 2024-01-02 | d | 0.7999999999999989 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

Each date cross-section is calculated independently, so values from other dates cannot influence the result.
