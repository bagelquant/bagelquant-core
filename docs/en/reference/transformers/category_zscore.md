# category_zscore

```python
category_zscore(source, category, name=None, metadata=None)
```

Apply `category_zscore` to long-form panel inputs.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**category** : Panel | Graph
: `category` argument.
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

from bagelquant_core import CategoryPanel, Domain, Panel
from bagelquant_core.transformer import category_zscore

domain = Domain(calendar=["2024-01-02"], universe=["a", "b", "c"])
factor = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": [1.0, 3.0, 8.0],
    }),
    domain,
)
industry = CategoryPanel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": ["tech", "tech", "bank"],
    }),
    domain,
)

result = category_zscore(factor, industry).compute().data
print(result)
```

## Notes

Inputs are long-form panels keyed by `(time, asset_id)`.

Missing group labels are excluded from the group calculation.
