# mask

```python
mask(frame, mask_frame, replace_value=nan, name=None, metadata=None)
```

Replace values where a mask frame is false.

## Parameters

**frame** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**mask_frame** : Panel | Graph
: Mask input. Truthy cells retain values; false or missing cells are replaced.
**replace_value** : float, default `nan`
: Value inserted where the mask is false or missing.
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
from bagelquant_core.composer import mask

left = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))
right = Panel(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}))

result = mask(left, right, replace_value=0).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.
