# Transformer

## Overview

A transformer is a unary function-style operation:

```text
Panel | Graph -> Graph
```

## Built-In Transformers

```python
from bagelquant_core.transformer import (
    rank,
    rolling_mean,
    signed_log1p,
    winsorize,
    zscore,
)

factor = rank(zscore(winsorize(raw_factor)), name="factor")
smoothed = rolling_mean(factor, window=20, name="smoothed")
compressed = signed_log1p(smoothed, name="compressed")
```

Built-ins are grouped by behavior:

| Family | Transformers |
| --- | --- |
| Basic | `identity`, `abs_value`, `negate`, `diff` |
| Rolling | `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, `rolling_sum` |
| Power | `power`, `signed_power`, `sqrt` |
| Logarithmic | `log`, `log1p`, `signed_log1p` |
| Normalization | `rank`, `zscore`, `winsorize`, `min_max_scale` |

Rolling and `diff` operations run over rows, which represent time. Normalization
operations run across columns, which represent assets. Logarithmic operations
and `sqrt` return `NaN` where their mathematical domain is undefined.

## User-Defined Transformers

```python
import pandas as pd

from bagelquant_core.transformer import transformer


@transformer
def demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


centered = demean(price, name="centered")
```

The decorated function receives a `DataFrame` during execution but accepts a
`Panel` or `Graph` when researchers construct a workflow.

Configuration arguments are stored in graph specifications and cache keys.
