# weighted_sum

```python
weighted_sum(*frames, weights, name=None, metadata=None)
```

Return the weighted sum of one or more frames.

## Parameters

**frames** : Panel | Graph
: One or more numeric `Panel` or single-output `Graph` inputs.
**weights** : Sequence[Real]
: One numeric weight for each input frame.
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

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import weighted_sum

domain = Domain(calendar=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), universe=["a", "b"])
left = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)
right = Panel.from_domain(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}, index=domain.sessions), domain)

result = weighted_sum(left, right, weights=[0.25, 0.75]).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.
