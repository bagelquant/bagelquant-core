# Composer

## Overview

A composer is a multi-input function-style operation:

```text
(Panel | Graph, ...) -> Graph
```

## Built-In Composers

```python
from bagelquant_core.composer import div, weighted_sum

ratio = div(book, price, name="bm_ratio")
prediction = weighted_sum(
    value,
    quality,
    momentum,
    weights=[0.4, 0.3, 0.3],
    name="prediction",
)
```

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

`weighted_sum(...)` requires one weight per input frame and computes a new
frame without mutating its inputs.
