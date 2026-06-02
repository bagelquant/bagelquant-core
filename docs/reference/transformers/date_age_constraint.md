# date_age_constraint

```python
date_age_constraint(source, window, min_valid=None, name=None, metadata=None)
```

Mask values until enough valid observations exist in a trailing window.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
**min_valid** : int | None, default `None`
: Minimum valid observations required within the trailing window.
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
from bagelquant_core.transformer import date_age_constraint

domain = Domain(region="US", universe=["a", "b"], start_date="2024-01-02", end_date="2024-01-04")
source = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)

result = date_age_constraint(source, window=2).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
