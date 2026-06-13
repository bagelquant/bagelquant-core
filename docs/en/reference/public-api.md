# Public API

The stable public API is exported from `bagelquant_core`,
`bagelquant_core.transformer`, and `bagelquant_core.composer`.

## Top-Level Objects

- `Domain`: trading sessions plus static or dynamic asset membership.
- `Panel`: immutable numeric long-form data aligned to a `Domain`.
- `CategoryPanel`: immutable long-form label data aligned to a `Domain`.
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
[`reference/transformers/index.md`](transformers/index.md).

## Composers

Composers accept one or more `Panel` or `Graph` inputs and return a `Graph`.

```python
from bagelquant_core.composer import div, weighted_sum

ratio = div(book, price, name="book_to_price")
prediction = weighted_sum(ratio, quality, weights=[0.6, 0.4])
```

The generated composer reference is in
[`reference/composers/index.md`](composers/index.md).

## Custom Operations

Use decorators when project-specific logic should behave like built-in
operations.

```python
import polars as pl

from bagelquant_core.composer import composer
from bagelquant_core.transformer import transformer


@transformer
def demean(frame: pl.DataFrame) -> pl.DataFrame:
    means = frame.group_by("time").agg(pl.col("value").mean().alias("mean"))
    return (
        frame.join(means, on="time")
        .with_columns((pl.col("value") - pl.col("mean")).alias("value"))
        .select("time", "asset_id", "value")
    )


@composer
def average(*frames: pl.DataFrame) -> pl.DataFrame:
    stacked = pl.concat(frames)
    return (
        stacked.group_by("time", "asset_id")
        .agg(pl.col("value").mean().alias("value"))
        .sort("time", "asset_id")
    )
```

## Compatibility Boundary

Public APIs are Polars DataFrame and `Panel` oriented. `bagelquant-core` does
not own data retrieval, provider credentials, persistence, portfolio
simulation, or application UI.

