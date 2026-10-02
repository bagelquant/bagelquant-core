# `rolling_elastic_net_prediction`

Fit and predict RMS-scaled Elastic Net using only matured historical labels.

## Signature

```python
rolling_elastic_net_prediction(*features, labels, label_end=None, label_available=None, window=252, fit_every=20, min_samples=200, max_samples=100000, label_maturity=2, anchor_offset=0, alpha=0.001, l1_ratio=0.5, max_iter=10000, tolerance=1e-06, seed=1729, name=None, metadata=None)
```

## Parameters

**features** : Panel | Graph
: One or more numeric feature Panels in fixed positional order.
**labels** : Panel | Graph
: Explicit auxiliary training-label Panel; only matured labels may be used.
**label_end** : pl.DataFrame | None, default `None`
: Optional auxiliary Panel of Gregorian date ordinals; each selected label must end by the fit date.
**label_available** : pl.DataFrame | None, default `None`
: Optional auxiliary Panel of Gregorian date ordinals; missing or future availability makes a row ineligible.
**window** : int, default `252`
: Positive trailing-window length in rows.
**fit_every** : int, default `20`
: Trading sessions between model fits, anchored to Data Start.
**min_samples** : int, default `200`
: Minimum number of complete mature training samples.
**max_samples** : int, default `100000`
: Maximum complete mature samples, allocated by date-balanced stable seeded keys before gathering features; part of DSL identity.
**label_maturity** : int, default `2`
: Sessions before a label is fully known; supplied from its auxiliary contract.
**anchor_offset** : int, default `0`
: Application supplied position of the first input session in the global calendar.
**alpha** : float, default `0.001`
: Smoothing or regularization parameter, depending on the operation.
**l1_ratio** : float, default `0.5`
: Elastic-net mixing parameter in `[0, 1]`.
**max_iter** : int, default `10000`
: Maximum coordinate-descent iterations.
**tolerance** : float, default `1e-06`
: Convergence tolerance for coordinate descent.
**seed** : int, default `1729`
: Deterministic numerical sampling seed.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
rolling_elastic_net_prediction(source, labels=labels, window=3, fit_every=1, min_samples=2, max_samples=100, label_maturity=1)
```

The call and tables below come from one deterministic, hand-checkable fixture.
Tables are pivoted wide only for readability; runtime Panels remain long-form.
`missing` is the canonical rendered form of null or mathematically invalid output.

### source

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 2 |
| 2024-01-03 | 2 | 3 |
| 2024-01-04 | 4 | 5 |
| 2024-01-05 | 7 | 8 |
### Panel parameter: labels

| time | a | b |
|---|---:|---:|
| 2024-01-02 | 1 | 1 |
| 2024-01-03 | 1.5 | 2 |
| 2024-01-04 | 2 | 2.5 |
| 2024-01-05 | 3 | 4 |

### Output

| time | a | b |
|---|---:|---:|
| 2024-01-02 | missing | missing |
| 2024-01-03 | 1 | 1 |
| 2024-01-04 | 2.36629 | 2.86194 |
| 2024-01-05 | 3.26127 | 3.64397 |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
