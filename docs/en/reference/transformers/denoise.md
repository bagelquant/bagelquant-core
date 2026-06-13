# denoise

```python
denoise(source, threshold=1e-12, name=None, metadata=None)
```

Apply `denoise` to long-form panel inputs.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**threshold** : float, default `1e-12`
: Non-negative magnitude below which values are replaced with zero.
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
from bagelquant_core.transformer import denoise

domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
source = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }),
    domain,
)

result = denoise(source, threshold=1e-6).compute().data
print(result)
```

## Notes

Inputs are long-form panels keyed by `(time, asset_id)`.
