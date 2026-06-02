# denoise

```python
denoise(source, threshold=1e-12, name=None, metadata=None)
```

Replace values whose absolute magnitude is tiny with zero.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**threshold** : float, default `1e-12`
: Non-negative magnitude below which values are replaced with zero.
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
from bagelquant_core.transformer import denoise

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = denoise(source, threshold=1e-6).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
