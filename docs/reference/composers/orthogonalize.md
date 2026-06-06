# orthogonalize

```python
orthogonalize(frame, *factors, name=None, metadata=None)
```

Return row-wise residuals after projecting values onto factors.

## Parameters

**frame** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**factors** : Panel | Graph
: One or more factor `Panel` or single-output `Graph` inputs.
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
from bagelquant_core.composer import orthogonalize

domain = Domain(calendar=pd.to_datetime(["2024-01-02"]), universe=["a", "b", "c"])
factor = Panel.from_domain(pd.DataFrame({"a": [1.0], "b": [3.0], "c": [5.0]}, index=domain.sessions), domain)
size = Panel.from_domain(pd.DataFrame({"a": [0.0], "b": [1.0], "c": [2.0]}, index=domain.sessions), domain)

result = orthogonalize(factor, size).compute().data
print(result)
```

## Notes

Inputs are aligned by index and columns before the operation runs.
