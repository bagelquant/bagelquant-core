# trim_quantile

```python
trim_quantile(source, lower=0.01, upper=0.99, name=None, metadata=None)
```

Replace row values outside cross-sectional quantile bounds with NaN.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**lower** : float, default `0.01`
: Lower fixed bound or lower quantile, depending on the operation.
**upper** : float, default `0.99`
: Upper fixed bound or upper quantile, depending on the operation.
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
from bagelquant_core.transformer import trim_quantile

domain = Domain(region="US", universe=["a", "b"], start_date="2024-01-02", end_date="2024-01-04")
source = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)

result = trim_quantile(source, lower=0.1, upper=0.9).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
