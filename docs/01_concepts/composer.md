# Composer

## Overview

A composer is a multi-input function-style operation:

```text
(Panel | Graph, ...) -> Graph
```

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
