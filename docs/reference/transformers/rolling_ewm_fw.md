# rolling_ewm_fw

```python
rolling_ewm_fw(source, halflife, min_periods=0, name=None, metadata=None)
```

Return expanding exponentially weighted means with a half-life.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**halflife** : float
: Half-life decay parameter. Supply exactly one decay parameter.
**min_periods** : int, default `0`
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
from bagelquant_core.transformer import rolling_ewm_fw

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = rolling_ewm_fw(source, halflife=2).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.

Rolling calculations run independently down each asset column.
