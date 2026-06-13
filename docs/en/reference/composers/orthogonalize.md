# orthogonalize

```python
orthogonalize(frame, *factors, name=None, metadata=None)
```

Apply `orthogonalize` to long-form panel inputs.

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
import polars as pl

from bagelquant_core import Domain, Panel
from bagelquant_core.composer import orthogonalize

domain = Domain(calendar=["2024-01-02"], universe=["a", "b", "c"])
factor = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": [1.0, 3.0, 5.0],
    }),
    domain,
)
size = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02"] * 3,
        "asset_id": ["a", "b", "c"],
        "value": [0.0, 1.0, 2.0],
    }),
    domain,
)

result = orthogonalize(factor, size).compute().data
print(result)
```

## Notes

Inputs are aligned by `(time, asset_id)` before the operation runs.
