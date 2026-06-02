# winsorize

```python
winsorize(source, lower=0.01, upper=0.99, name=None, metadata=None)
```

Clip each row to its lower and upper quantiles.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**lower** : float, default `0.01`
: Lower fixed bound or lower quantile, depending on the operation.
**upper** : float, default `0.99`
: Upper fixed bound or upper quantile, depending on the operation.
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
from bagelquant_core.transformer import winsorize

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = winsorize(source).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
