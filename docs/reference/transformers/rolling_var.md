# rolling_var

```python
rolling_var(source, window, min_periods=None, ddof=1, name=None, metadata=None)
```

Return rolling variances over time.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
**min_periods** : int | None, default `None`
: Minimum number of observations required to produce a value.
**ddof** : int, default `1`
: Delta degrees of freedom used by variance or standard-deviation calculations.
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
from bagelquant_core.transformer import rolling_var

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = rolling_var(source, window=2).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.

Rolling calculations run independently down each asset column.
