# Composer

## Overview

A composer is a multi-input function-style operation:

```text
(Panel | Graph, ...) -> Graph
```

For signatures, parameter descriptions, and examples for every public
operation, see the [composer reference](../reference/composers/index.md).

## Built-In Composers

```python
from bagelquant_core.composer import div, mean, weighted_mean

ratio = div(book, price, name="bm_ratio")
consensus = mean(value, quality, momentum, name="consensus")
prediction = weighted_mean(
    value,
    quality,
    momentum,
    weights=[0.4, 0.3, 0.3],
    name="prediction",
)
```

Built-ins are grouped by behavior:

| Family | Composers |
| --- | --- |
| Arithmetic | `add`, `sub`, `mul`, `div` |
| Aggregation | `sum_frames`, `mean`, `product`, `minimum`, `maximum`, `weighted_sum`, `weighted_mean` |
| General | `project`, `mask`, `coalesce` |
| Scaling | `vol_scale` |
| Math | `power`, `power_df`, `and_`, `or_`, `not_`, `xand`, `xor`, `greater`, `greater_equal`, `less`, `less_equal`, `equal` |
| Rolling | `rolling_corr`, `rolling_cov`, `rolling_ols`, `rolling_lasso`, `rolling_ridge`, `rolling_elastic_net` |
| Cross-sectional | `orthogonalize`, `group_rank`, `group_mean`, `group_max`, `group_min`, `group_median`, `group_std`, `group_demean`, `group_zscore`, `group_rankpct`, `group_percentile` |

## User-Defined Composers

```python
import pandas as pd

from bagelquant_core.composer import composer


@composer
def average(*frames: pd.DataFrame) -> pd.DataFrame:
    return sum(frames) / len(frames)


combined = average(value, quality, momentum, name="combined")
```

The internal execution runtime aligns input panel data before executing a
composer. Already-aligned inputs are reused internally.

Weighted composers require one numeric weight per input frame and compute a
new frame without mutating their inputs. `weighted_mean(...)` also requires a
non-zero total weight.

Rolling regressions use `rolling_ols(y, *factors, window=...)` and the same
input order for regularized variants. They fit on prior rows only, then predict
the current row.

Comparison and logical composers return numeric `1.0` and `0.0` panels so their
outputs remain valid graph inputs. `minimum` and `maximum` are also exported as
`min` and `max`; `sub`, `mul`, and `div` are exported as `subtract`,
`multiply`, and `divide`.
