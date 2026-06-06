# Public API

The stable public API is exported from `bagelquant_core`,
`bagelquant_core.transformer`, and `bagelquant_core.composer`.

## Top-Level Objects

- `Domain`: trading sessions plus static or dynamic asset membership.
- `Panel`: immutable numeric time-by-asset data aligned to a `Domain`.
- `CategoryPanel`: immutable time-by-asset label data aligned to a `Domain`.
- `Graph`: lazy derived logic produced by transformers and composers.

```python
from bagelquant_core import CategoryPanel, Domain, Graph, Panel
```

## Transformers

Transformers accept one `Panel` or `Graph` and return a `Graph`.

```python
from bagelquant_core.transformer import rank, winsorize, zscore

factor = rank(zscore(winsorize(raw_panel)), name="factor")
```

The generated transformer reference is in
[`docs/reference/transformers/index.md`](reference/transformers/index.md).

## Composers

Composers accept one or more `Panel` or `Graph` inputs and return a `Graph`.

```python
from bagelquant_core.composer import div, weighted_sum

ratio = div(book, price, name="book_to_price")
prediction = weighted_sum(ratio, quality, weights=[0.6, 0.4])
```

The generated composer reference is in
[`docs/reference/composers/index.md`](reference/composers/index.md).

## Custom Operations

Use decorators when project-specific logic should behave like built-in
operations.

```python
import pandas as pd

from bagelquant_core.composer import composer
from bagelquant_core.transformer import transformer


@transformer
def demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


@composer
def average(*frames: pd.DataFrame) -> pd.DataFrame:
    return sum(frames) / len(frames)
```

## Compatibility Boundary

Public APIs are DataFrame and `Panel` oriented. `bagelquant-core` does not own
data retrieval, provider credentials, persistence, portfolio simulation, or
application UI.

