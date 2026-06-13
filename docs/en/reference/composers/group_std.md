# group_std

```python
group_std(frame, group, name=None, metadata=None)
```

Replace each element with its row-wise group standard deviation.

## Parameters

**frame** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**group** : Panel | Graph
: Matching `CategoryPanel` containing row-wise group labels.
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
from bagelquant_core.composer import group_std

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

result = group_std(factor, industry).compute().data
print(result)
```

## Notes

Inputs are aligned by `(time, asset_id)` before the operation runs.

Missing group labels are excluded from the group calculation.
