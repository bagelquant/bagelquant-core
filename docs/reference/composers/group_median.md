# group_median

```python
group_median(frame, group, name=None, metadata=None)
```

Replace each element with its row-wise group median.

## Parameters

**frame** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**group** : Panel | Graph
: Matching `CategoryPanel` containing row-wise group labels.
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

from bagelquant_core import CategoryPanel, Panel
from bagelquant_core.composer import group_median

factor = Panel(pd.DataFrame({"a": [1.0], "b": [3.0], "c": [8.0]}))
industry = CategoryPanel(pd.DataFrame({"a": ["tech"], "b": ["tech"], "c": ["bank"]}))

result = group_median(factor, industry).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.

Missing group labels are excluded from the group calculation.
