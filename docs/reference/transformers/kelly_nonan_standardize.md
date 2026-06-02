# kelly_nonan_standardize

```python
kelly_nonan_standardize(source, window, name=None, metadata=None)
```

Fill missing values, standardize cross-sectionally, then estimate Kelly.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
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
from bagelquant_core.transformer import kelly_nonan_standardize

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = kelly_nonan_standardize(source, window=2).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
