# signed_power

```python
signed_power(source, exponent, name=None, metadata=None)
```

Raise absolute values to an exponent while preserving signs.

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
from bagelquant_core.transformer import signed_power

domain = Domain(calendar=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), universe=["a", "b"])
source = Panel.from_domain(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}, index=domain.sessions), domain)

result = signed_power(source, exponent=0.5).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.
