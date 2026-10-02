# `rolling_lightgbm_prediction`

Fit deterministic CPU LightGBM using explicit matured labels and resource budgets.

## Signature

```python
rolling_lightgbm_prediction(*features, labels, label_end=None, label_available=None, window=252, fit_every=20, min_samples=200, max_samples=100000, label_maturity=2, anchor_offset=0, num_leaves=31, learning_rate=0.05, min_data_in_leaf=20, lambda_l1=0.0, lambda_l2=1.0, feature_fraction=1.0, num_boost_round=100, max_bin=255, max_depth=-1, min_gain_to_split=0.0, seed=1729, name=None, metadata=None)
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
**num_leaves** : int, default `31`
: Maximum LightGBM leaves per tree.
**learning_rate** : float, default `0.05`
: LightGBM boosting learning rate.
**min_data_in_leaf** : int, default `20`
: Minimum training observations per leaf.
**lambda_l1** : float, default `0.0`
: LightGBM L1 leaf penalty.
**lambda_l2** : float, default `1.0`
: LightGBM L2 leaf penalty.
**feature_fraction** : float, default `1.0`
: Fraction of model features sampled per tree.
**num_boost_round** : int, default `100`
: Number of boosting rounds.
**max_bin** : int, default `255`
: Maximum histogram bins per feature.
**max_depth** : int, default `-1`
: Maximum tree depth; -1 is unlimited.
**min_gain_to_split** : float, default `0.0`
: Minimum improvement needed for a tree split.
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
rolling_lightgbm_prediction(source, labels=labels, window=3, fit_every=1, min_samples=2, max_samples=100, label_maturity=1, num_boost_round=2, min_data_in_leaf=1)
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
| 2024-01-04 | 1.40586 | 1.40586 |
| 2024-01-05 | 1.70346 | 1.70346 |

## Panel and temporal semantics

Peer inputs are aligned on `(time, asset_id)` before the operation runs; alignment does not invent Universe membership.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.
