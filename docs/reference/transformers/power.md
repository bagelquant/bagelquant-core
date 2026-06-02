# power

```python
power(source, exponent, name=None, metadata=None)
```

Raise each element of the first input to the corresponding element of the second input.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**exponent** : Real
: Numeric exponent.
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
from bagelquant_core.transformer import power

domain = Domain(region="US", universe=["a", "b"], start_date="2024-01-02", end_date="2024-01-04")
source = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)

result = power(source, exponent=2).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
