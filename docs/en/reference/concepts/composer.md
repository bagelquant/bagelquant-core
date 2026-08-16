# Composer

## Overview

A composer fuses at least two peer Panel inputs into one Panel:

```text
(Panel | Graph, Panel | Graph, ...) -> Graph[Panel]
```

For signatures, parameter descriptions, and examples for every public
operation, see the [composer reference](../composers/index.md).

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
| Missing data | `coalesce` |
| Logical and comparison | `and_`, `or_`, `xand`, `xor`, `greater`, `greater_equal`, `less`, `less_equal`, `equal` |
| Rolling relationships | `rolling_corr`, `rolling_cov` |

## User-Defined Composers

```python
import polars as pl

from bagelquant_core.composer import composer


@composer
def average(*frames: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.concat(frames)
        .group_by("time", "asset_id")
        .agg(pl.col("value").mean().alias("value"))
        .sort("time", "asset_id")
    )


combined = average(value, quality, momentum, name="combined")
```

The internal execution runtime aligns every peer input by `(time, asset_id)`
before executing a composer. Already-aligned inputs are reused internally. A
composer call with fewer than two inputs is invalid, including variable-input
operations such as `sum_frames`, `mean`, and `coalesce`.

Weighted composers require one numeric weight per input frame and compute a
new frame without mutating their inputs. `weighted_mean(...)` also requires a
non-zero total weight.

Operations with one semantic source plus an auxiliary Panel—grouping,
neutralization, masking, projection, volatility scaling, logical negation, and
rolling regression—are Transformers and use named Panel parameters. See the
[Transformer concept](./transformer.md).

Comparison and logical composers return numeric `1.0` and `0.0` panels so their
outputs remain valid graph inputs. Use the canonical `minimum`, `maximum`,
`sub`, `mul`, and `div` operation names in code and serialized graphs.
