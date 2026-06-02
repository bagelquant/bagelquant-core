# rolling_corr

```python
rolling_corr(lhs, rhs, window, min_periods=None, name=None, metadata=None)
```

Return rolling correlations between corresponding columns.

## Parameters

**lhs** : Panel | Graph
: Left-hand numeric `Panel` or single-output `Graph`.
**rhs** : Panel | Graph
: Right-hand numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
**min_periods** : int | None, default `None`
: Minimum number of observations required to produce a value.
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
from bagelquant_core.composer import rolling_corr

left = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))
right = Panel(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}))

result = rolling_corr(left, right, window=2).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.

Rolling calculations run independently down each asset column.
