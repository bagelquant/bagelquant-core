# Concepts

`bagelquant-core` models research logic as domain-aware data plus lazy
operation graphs.

## Domain

A `Domain` defines the sessions and asset universe for a research problem. It
does not download calendars or security masters; callers provide the calendar
and either a static asset list or a dynamic membership frame.

## Panel

A `Panel` is immutable time-by-asset numeric data aligned to a `Domain`.
Panels are raw inputs, graph leaves, materialized graph outputs, and cache
values during execution.

See [Panel](reference/concepts/panel.md).

`PredictionPanel` is the strong subtype produced by a `PredictionComposer`.
Transformers may post-process its semantic input and preserve that type;
ordinary composers and auxiliary Panel parameters cannot consume predictions.
Ordinary composers continue to produce `Panel` AlphaValues.

## Graph

A `Graph` is a lazy chain of research logic. Transformers and composers return
graphs, and execution materializes output panels only when `compute()` is
called.

See [Graph](reference/concepts/graph.md).

## Transformers

Transformers are unary operations:

```text
Panel | Graph -> Graph
```

They cover ranking, normalization, rolling windows, missing-value handling,
outlier treatment, and other common factor transformations.

See [Transformer](reference/concepts/transformer.md).

## Composers

Composers combine multiple `Panel` or `Graph` inputs into one graph. They cover
arithmetic, cross-sectional grouping, rolling relationships, projection, and
weighted aggregation.

See [Composer](reference/concepts/composer.md).

## Prediction composers

Prediction construction is an allowlisted typed graph operation. Identity,
equal-weight, rolling positive-IC, exponentially decayed rolling positive-IC,
rolling positive quantile-rank-IC, rolling OLS, and rolling GLS composers consume
AlphaValue Panels already aligned and standardized by the caller's Alpha Policy.
Supervised composers receive generic target and availability Panels; core does
not own prices, schedules, standardization, or backtest policies. Window length
is measured in periods prepared by the caller, and missing values are never
forward-filled.

A PredictionComposer output may feed a Transformer chain. Each Transformer
keeps the `PredictionPanel` boundary, but predictions cannot enter another
Composer or be supplied as a Transformer's named Panel parameter.

`QuantileICWeightedPredictionComposer(window, quantiles)` groups every finite
Alpha cross section before target missingness is considered, computes Spearman
correlation between high-to-low quantile scores and quantile mean targets, and
uses the positive part of the complete-window mean IC. Its graph identity
serializes both `window` and `quantiles`.

`ICWeightedDecayPredictionComposer(window, half_life)` applies exponentially
decaying period weights to the same complete contiguous Spearman-IC window.
For observations ordered oldest to newest, period `i` receives weight
`2 ** (-(window - 1 - i) / half_life)`. The positive weighted mean becomes the
Alpha weight, and graph identity serializes both `window` and `half_life`.

For application-orchestrated learning, `ElasticNetPredictionComposer` serializes
the complete walk-forward, coverage, target, validation, and Elastic Net
contracts without depending on a backtest package. `ZeroPreservingRmsScaler`
fits weighted RMS scales without centering, so gated zeros remain zeros.
Relative regularization candidates are computed from each fold's sample-only
`alpha_max`; a sample-plus-validation refit recomputes `alpha_max` while keeping
the selected ratio.

## Execution

Execution validates the graph, computes dependencies in topological order,
caches intermediate outputs, and applies domain membership rules so inactive
cells do not leak into later operations.

See [Execution](reference/concepts/execution.md).
