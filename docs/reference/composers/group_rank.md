# group_rank

```python
group_rank(frame, group, name=None, metadata=None)
```

Return row-wise ranks within each group.

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
from bagelquant_core.composer import group_rank

factor = Panel(pd.DataFrame({"a": [1.0], "b": [3.0], "c": [8.0]}))
industry = CategoryPanel(pd.DataFrame({"a": ["tech"], "b": ["tech"], "c": ["bank"]}))

result = group_rank(factor, industry).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.

Missing group labels are excluded from the group calculation.
