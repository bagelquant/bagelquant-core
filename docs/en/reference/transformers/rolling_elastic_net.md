# `rolling_elastic_net`

Predict the current target from factors fitted only on prior per-asset rows with elastic-net regularization.

## Signature

```python
rolling_elastic_net(source, *, factors, window, alpha=1.0, l1_ratio=0.5, max_iter=1000, tolerance=1e-08, name=None, metadata=None)
```

## Parameters

**target** : Panel | Graph
: Dependent-variable `Panel` predicted from the trailing factor window.
**factors** : tuple[pl.DataFrame, ...]
: One or more factor `Panel` or single-output `Graph` inputs.
**window** : int
: Positive trailing-window length in rows.
**alpha** : float, default `1.0`
: Smoothing or regularization parameter, depending on the operation.
**l1_ratio** : float, default `0.5`
: Elastic-net mixing parameter in `[0, 1]`.
**max_iter** : int, default `1000`
: Maximum coordinate-descent iterations.
**tolerance** : float, default `1e-08`
: Convergence tolerance for coordinate descent.
**name** : str | None, default `None`
: Optional graph-node name. A generated name is used when omitted.
**metadata** : Mapping[str, Any] | None, default `None`
: Optional metadata stored on the graph node.

## Returns

**Graph**
: Lazy single-output graph. Call `.compute()` to materialize a `Panel`.

## Executable Panel example

```python
rolling_elastic_net(source, factors=(factors,), window=2)
```

The call and tables below come from one deterministic, hand-checkable fixture.
`missing` is the canonical rendered form of null or mathematically invalid
output.

### source

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 2.0 |
| 2024-01-03 | a | 2.0 |
| 2024-01-03 | b | 3.0 |
| 2024-01-04 | a | 4.0 |
| 2024-01-04 | b | 5.0 |
| 2024-01-05 | a | 7.0 |
| 2024-01-05 | b | 8.0 |
### Panel parameter: factors

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | 1.0 |
| 2024-01-02 | b | 1.0 |
| 2024-01-03 | a | 1.5 |
| 2024-01-03 | b | 2.0 |
| 2024-01-04 | a | 2.0 |
| 2024-01-04 | b | 2.5 |
| 2024-01-05 | a | 3.0 |
| 2024-01-05 | b | 4.0 |

### Output

| time | asset_id | value |
|---|---|---:|
| 2024-01-02 | a | missing |
| 2024-01-02 | b | missing |
| 2024-01-03 | a | missing |
| 2024-01-03 | b | missing |
| 2024-01-04 | a | 1.5 |
| 2024-01-04 | b | 2.5 |
| 2024-01-05 | a | 3.0 |
| 2024-01-05 | b | 4.0 |

## Panel and temporal semantics

The primary input is one sparse long-form Panel keyed by `(time, asset_id)`; absent keys remain absent.

History is grouped by `asset_id` and ordered by `time`; rows with insufficient observations remain missing.

The model is fit on prior rows only and predicts the current row.
