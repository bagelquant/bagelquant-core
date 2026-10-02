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

## Prediction Composers

Typed prediction construction is exported from `bagelquant_core`. A resulting
`PredictionPanel` may pass through Transformer chains while retaining its type,
but it cannot feed another Composer or a named Panel parameter. The
quantile-rank implementation and its composer share one public numerical
contract:

```python
from bagelquant_core import (
    ICWeightedDecayPredictionComposer,
    QuantileICWeightedPredictionComposer,
    quantile_rank_information_coefficient,
)

composer = QuantileICWeightedPredictionComposer(window=12, quantiles=10)
decayed = ICWeightedDecayPredictionComposer(window=12, half_life=6)
```

`q1` contains the highest Alpha values. A strictly decreasing q1-to-qN target
return path has quantile rank IC `+1`; incomplete groups or equal group means
produce no IC.

The decayed composer uses the same complete-window and positive-IC rules while
giving each IC observation half the newest observation's weight after
`half_life` caller-supplied periods.

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

## Causal Prediction Processing

`smooth_prediction(prediction, config, evaluation_calendar=..., state=None)`
accepts only `PredictionPanel` and returns `PredictionSmoothingResult` with
`.prediction` and `.state`. Configure `PredictionSmoothingConfig(method="none")`,
`PredictionSmoothingConfig(method="sma", window=3)`, or
`PredictionSmoothingConfig(method="ewma", half_life=2)`.

The calendar is strictly ordered and unique. Windows count evaluation periods
(daily sessions or monthly evaluations), not elapsed days. SMA requires a full
continuous finite window; EWMA initializes at the first finite observation and
uses `alpha = 1 - 2**(-1/half_life)`. Missing values, infinity and Universe exits
reset each asset independently. No value is filled or carried forward.

The input Domain must contain a contiguous block of the supplied calendar.
Fresh calls require the complete history from the calendar's first date;
resumes require a matching checkpoint immediately before the block. Persist
checkpoints with `state.to_dict()` and `PredictionSmoothingState.from_dict(...)`.
Calendar-prefix and configuration validation rejects invalid resume boundaries.
Changing a visible research window must not restart smoothing. This public
numerical helper is not a registered unrestricted graph/DSL operation.

## Compatibility Boundary

`WeightedRegressionMoments.scale_weights(factor)` rescales every accumulated
weighted sum by a positive finite scalar while preserving `observation_count`.
It mutates only that accumulator; earlier `copy()` snapshots stay independent.
Streaming decay consumers can rescale prior moments before adding the next
batch, keeping the reference at the latest observed batch without reading a
future endpoint. Workbench owns the decay policy and training-period selection.

Public APIs are Polars DataFrame and `Panel` oriented. `bagelquant-core` does
not own data retrieval, provider credentials, persistence, portfolio
simulation, or application UI.

