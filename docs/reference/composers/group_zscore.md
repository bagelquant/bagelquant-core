# group_zscore

```python
group_zscore(frame, group, name=None, metadata=None)
```

Return row-wise z-scores within each group.

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

from bagelquant_core import CategoryPanel, Domain, Panel
from bagelquant_core.composer import group_zscore

domain = Domain(region="US", universe=["a", "b", "c"], start_date="2024-01-02", end_date="2024-01-02")
factor = Panel.from_domain(pd.DataFrame({"a": [1.0], "b": [3.0], "c": [8.0]}, index=domain.sessions), domain)
industry = CategoryPanel.from_domain(pd.DataFrame({"a": ["tech"], "b": ["tech"], "c": ["bank"]}, index=domain.sessions), domain)

result = group_zscore(factor, industry).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.

Missing group labels are excluded from the group calculation.
