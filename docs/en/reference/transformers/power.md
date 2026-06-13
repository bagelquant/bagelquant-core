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
import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.transformer import power

domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
source = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }),
    domain,
)

result = power(source, exponent=2).compute().data
print(result)
```

## Notes

Inputs are long-form panels keyed by `(time, asset_id)`.
