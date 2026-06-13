# rolling_corr

```python
rolling_corr(lhs, rhs, window, min_periods=None, name=None, metadata=None)
```

Apply `rolling_corr` to long-form panel inputs.

## Parameters

**lhs** : Panel | Graph
: Left-hand numeric `Panel` or single-output `Graph`.
**rhs** : Panel | Graph
: Right-hand numeric `Panel` or single-output `Graph`.
**window** : int
: Positive trailing-window length in rows.
**min_periods** : int | None, default `None`
: Minimum number of observations required to produce a value.
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
from bagelquant_core.composer import rolling_corr

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

result = rolling_corr(left, right, window=2).compute().data
print(result)
```

## Notes

Inputs are aligned by `(time, asset_id)` before the operation runs.

Rolling calculations run independently for each `asset_id` ordered by `time`.
