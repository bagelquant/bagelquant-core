# fillna

```python
fillna(source, value=0, name=None, metadata=None)
```

Fill missing values with a numeric scalar.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**value** : float, default `0`
: Numeric replacement or constant value.
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
from bagelquant_core.transformer import fillna

domain = Domain(region="US", universe=["a", "b"], start_date="2024-01-02", end_date="2024-01-04")
source = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)

result = fillna(source, value=0).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
