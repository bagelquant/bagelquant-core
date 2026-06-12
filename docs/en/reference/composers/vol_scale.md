# vol_scale

```python
vol_scale(frame, volatility, name=None, metadata=None)
```

Scale values by volatility.

## Parameters

**frame** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**volatility** : Panel | Graph
: Volatility `Panel` or single-output `Graph` used as the divisor.
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
from bagelquant_core.composer import vol_scale

domain = Domain(calendar=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), universe=["a", "b"])
left = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)
right = Panel.from_domain(pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 4.0]}, index=domain.sessions), domain)

result = vol_scale(left, right).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.
