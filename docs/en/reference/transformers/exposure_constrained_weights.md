# `exposure_constrained_weights`

Optimize target weights within explicit exposure bounds using historical target state.

## Signature

```python
exposure_constrained_weights(source, *, exposures, lower_bounds, upper_bounds, concentration_penalty=10.0, turnover_penalty=0.1, max_weight=0.04, max_turnover=None, constraint_tolerance=1e-07, reference=None, name=None, metadata=None)
```

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**exposures** : tuple[pl.DataFrame, ...]
: Explicit auxiliary risk Panels, ordered like the bounds; missing required exposure fails.
**lower_bounds** : list[float]
: One lower portfolio exposure bound per auxiliary risk Panel.
**upper_bounds** : list[float]
: One upper portfolio exposure bound per auxiliary risk Panel.
**concentration_penalty** : float, default `10.0`
: Non-negative quadratic penalty on target-weight concentration.
**turnover_penalty** : float, default `0.1`
: Non-negative penalty against the previous computed target weights.
**max_weight** : float, default `0.04`
: Maximum target weight per asset; insufficient capacity is unavailable.
**max_turnover** : float | None, default `None`
: Optional total absolute target-weight change limit.
**constraint_tolerance** : float, default `1e-07`
: Numerical solver feasibility tolerance; part of research identity.
**reference** : pl.DataFrame | None, default `None`
: Optional historical computed target Panel; never account holdings.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
exposure_constrained_weights(source, exposures=(exposures,), max_weight=1.0, lower_bounds=[-100.0], upper_bounds=[100.0])
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 4 | 3 |
| 2024-01-03 | missing | 2 | 8 |
### Panel parameter: exposures

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 1 | 2 | 2 |
| 2024-01-03 | 2 | 1 | 4 |

### Output

| time | a | b | c |
|---|---:|---:|---:|
| 2024-01-02 | 0.25 | 0.4 | 0.35 |
| 2024-01-03 | 0 | 0.355 | 0.645 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.
