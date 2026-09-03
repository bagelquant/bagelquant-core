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
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 3 | 6 | 10 |
### Panel parameter: factors

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | 1 | 1 | 2 | 3 |

### Output

| time | a | b | c | d |
|---|---:|---:|---:|---:|
| 2024-01-02 | -2.06667 | -0.0666667 | -0.133333 | 0.8 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

Each date cross-section is calculated independently, so values from other dates cannot influence the result.
