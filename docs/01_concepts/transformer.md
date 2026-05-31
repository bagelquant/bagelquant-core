# Transformer

## Overview

A transformer is a unary function-style operation:

```text
Panel | Graph -> Graph
```

## Built-In Transformers

```python
from bagelquant_core.transformer import rank, winsorize, zscore

factor = rank(zscore(winsorize(raw_factor)), name="factor")
```

## User-Defined Transformers

```python
import pandas as pd

from bagelquant_core.transformer import transformer


@transformer
def rolling_mean(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    return frame.rolling(window).mean()


smoothed = rolling_mean(price, window=20, name="smoothed")
```

The decorated function receives a `DataFrame` during execution but accepts a
`Panel` or `Graph` when researchers construct a workflow.

Configuration arguments are stored in graph specifications and cache keys.
