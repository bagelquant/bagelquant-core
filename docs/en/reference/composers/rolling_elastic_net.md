# rolling_elastic_net

```python
rolling_elastic_net(target, factor, window, alpha=1.0, l1_ratio=0.5, name=None, metadata=None)
```

Apply `rolling_elastic_net` to long-form panel inputs.

## Parameters

**target** : Panel | Graph
: `target` argument.
**factor** : Panel | Graph
: `factor` argument.
**window** : int
: Positive trailing-window length in rows.
**alpha** : float, default `1.0`
: Smoothing or regularization parameter, depending on the operation.
**l1_ratio** : float, default `0.5`
: Elastic-net mixing parameter in `[0, 1]`.
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
