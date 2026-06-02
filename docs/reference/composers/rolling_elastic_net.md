# rolling_elastic_net

```python
rolling_elastic_net(y, *factors, window, alpha=1.0, l1_ratio=0.5, max_iter=1000, tolerance=1e-08, name=None, metadata=None)
```

Return prior-window elastic-net predictions.

## Parameters

**y** : Panel | Graph
: Dependent-variable `Panel` or single-output `Graph`.
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
import pandas as pd

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import rolling_elastic_net

domain = Domain(region="US", universe=["a", "b"], start_date="2024-01-02", end_date="2024-01-04")
left = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)
right = Panel.from_domain(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}, index=domain.sessions), domain)

result = rolling_elastic_net(left, right, window=2).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.

Rolling calculations run independently down each asset column.

The model is fit on prior rows only and predicts the current row.
