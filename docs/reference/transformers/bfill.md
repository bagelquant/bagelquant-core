# bfill

```python
bfill(source, limit=None, name=None, metadata=None)
```

Backward-fill missing values over time.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**limit** : int | None, default `None`
: Maximum number of consecutive missing values to fill.
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
from bagelquant_core.transformer import bfill

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = bfill(source).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
