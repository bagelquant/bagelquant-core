# mask

```python
mask(frame, mask_frame, replace_value=nan, name=None, metadata=None)
```

Apply `mask` to long-form panel inputs.

## Parameters

**frame** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**mask_frame** : Panel | Graph
: Mask input. Truthy cells retain values; false or missing cells are replaced.
**replace_value** : float, default `nan`
: Value inserted where the mask is false or missing.
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
from bagelquant_core.composer import mask

domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
left = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }),
    domain,
)
right = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 1.0, 2.0, 1.0, 2.0, 4.0],
    }),
    domain,
)

result = mask(left, right, replace_value=0.0).compute().data
print(result)
```

## Notes

Inputs are aligned by `(time, asset_id)` before the operation runs.
