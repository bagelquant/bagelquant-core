# subtract

```python
subtract(lhs, rhs, name=None, metadata=None)
```

Alias for [`sub`](./sub.md). Subtract the second input from the first element-wise.

## Parameters

**lhs** : Panel | Graph
: Left-hand numeric `Panel` or single-output `Graph`.
**rhs** : Panel | Graph
: Right-hand numeric `Panel` or single-output `Graph`.
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
from bagelquant_core.composer import subtract

left = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))
right = Panel(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}))

result = subtract(left, right).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.
