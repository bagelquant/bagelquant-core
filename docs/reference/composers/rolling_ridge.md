# rolling_ridge

```python
rolling_ridge(y, *factors, window, alpha=1.0, name=None, metadata=None)
```

Return prior-window ridge-regression predictions.

## Parameters

**y** : Panel | Graph
: Dependent-variable `Panel` or single-output `Graph`.
**factors** : Panel | Graph
: One or more factor `Panel` or single-output `Graph` inputs.
**window** : int
: Positive trailing-window length in rows.
**alpha** : float, default `1.0`
: Smoothing or regularization parameter, depending on the operation.
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

from bagelquant_core import Panel
from bagelquant_core.composer import rolling_ridge

left = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))
right = Panel(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}))

result = rolling_ridge(left, right, window=2).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.

Rolling calculations run independently down each asset column.

The model is fit on prior rows only and predicts the current row.
