# replace_non_nan

```python
replace_non_nan(source, value, name=None, metadata=None)
```

Replace existing non-missing values with a numeric scalar.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**value** : Real
: Numeric replacement or constant value.
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
from bagelquant_core.transformer import replace_non_nan

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = replace_non_nan(source, value=1).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
