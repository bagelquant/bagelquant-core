# rolling_ewm

```python
rolling_ewm(source, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, name=None, metadata=None)
```

Alias for [`ewm_mean`](./ewm_mean.md). Return exponentially weighted means.

## Parameters

**source** : Panel | Graph
: Input numeric `Panel` or single-output `Graph`.
**com** : float | None, default `None`
: Center-of-mass decay parameter. Supply exactly one decay parameter.
**span** : float | None, default `None`
: Span decay parameter. Supply exactly one decay parameter.
**halflife** : float | None, default `None`
: Half-life decay parameter. Supply exactly one decay parameter.
**alpha** : float | None, default `None`
: Smoothing or regularization parameter, depending on the operation.
**min_periods** : int, default `0`
: Minimum number of observations required to produce a value.
**adjust** : bool, default `True`
: Whether to divide by the decaying adjustment factor.
**ignore_na** : bool, default `False`
: Whether missing values are ignored when calculating weights.
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
from bagelquant_core.transformer import rolling_ewm

domain = Domain(calendar=["2024-01-02", "2024-01-03", "2024-01-04"], universe=["a", "b"])
source = Panel.from_domain(
    pl.DataFrame({
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 2,
        "asset_id": ["a"] * 3 + ["b"] * 3,
        "value": [1.0, 2.0, 4.0, 2.0, 3.0, 8.0],
    }),
    domain,
)

result = rolling_ewm(source, span=2).compute().data
print(result)
```

## Notes

Inputs are long-form panels keyed by `(time, asset_id)`.

Rolling calculations run independently for each `asset_id` ordered by `time`.
