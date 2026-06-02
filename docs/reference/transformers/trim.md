# trim

```python
trim(source, lower, upper, name=None, metadata=None)
```

Replace values outside fixed lower and upper bounds with NaN.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**lower** : Real
: Lower fixed bound or lower quantile, depending on the operation.
**upper** : Real
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
from bagelquant_core.transformer import trim

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = trim(source, lower=-1, upper=1).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
