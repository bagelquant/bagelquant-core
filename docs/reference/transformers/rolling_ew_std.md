# rolling_ew_std

```python
rolling_ew_std(source, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, bias=False, name=None, metadata=None)
```

Alias for [`ewm_std`](./ewm_std.md). Return exponentially weighted standard deviations.

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
**bias** : bool, default `False`
: Whether to use the biased exponentially weighted variance calculation.
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

from bagelquant_core import Panel
from bagelquant_core.transformer import rolling_ew_std

source = Panel(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 8.0]}))

result = rolling_ew_std(source, span=2).compute().data
print(result)
```

## Notes

Rows represent time and columns represent assets.

Rolling calculations run independently down each asset column.
