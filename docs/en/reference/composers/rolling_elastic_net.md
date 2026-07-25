# rolling_elastic_net

```python
rolling_elastic_net(target, *factors, window, alpha=1.0, l1_ratio=0.5, max_iter=1000, tolerance=1e-08, name=None, metadata=None)
```

Predict the current target from factors fitted on the prior window with elastic-net regularization.

## Parameters

**target** : Panel | Graph
: `target` argument.
**factors** : Panel | Graph
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

## Examples

```python
import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import rolling_elastic_net

domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
left = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }),
    domain,
)
right = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 1.0, 2.0, 1.0, 2.0, 4.0],
    }),
    domain,
)

result = rolling_elastic_net(left, right, window=2).compute().data
print(result)
```

## Notes

Inputs are aligned by `(time, asset_id)` before the operation runs.

Rolling calculations run independently for each `asset_id` ordered by `time`.

The model is fit on prior rows only and predicts the current row.
