# rolling_var

```python
rolling_var(source, window, min_periods=None, ddof=1, name=None, metadata=None)
```

Apply `rolling_var` to long-form panel inputs.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
**min_periods** : int | None, default `None`
: Minimum number of observations required to produce a value.
**ddof** : int, default `1`
: Delta degrees of freedom used by variance or standard-deviation calculations.
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
from bagelquant_core.transformer import rolling_var

domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
source = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }),
    domain,
)

result = rolling_var(source, window=2).compute().data
print(result)
```

## Notes

Inputs are long-form panels keyed by `(time, asset_id)`.

Rolling calculations run independently for each `asset_id` ordered by `time`.
