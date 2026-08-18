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

`PredictionPanel` is the strong terminal subtype produced only by a
`PredictionComposer`. Ordinary composers always produce ordinary `Panel` values,
which represent AlphaValues rather than predictions.

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

Prediction construction is an allowlisted terminal graph operation. Identity,
equal-weight, rolling positive-IC, rolling positive quantile-rank-IC, rolling
OLS, and rolling GLS composers consume
AlphaValue Panels already aligned and standardized by the caller's Alpha Policy.
Supervised composers receive generic target and availability Panels; core does
not own prices, schedules, standardization, or backtest policies. Window length
is measured in periods prepared by the caller, and missing values are never
forward-filled.

`QuantileICWeightedPredictionComposer(window, quantiles)` groups every finite
Alpha cross section before target missingness is considered, computes Spearman
correlation between high-to-low quantile scores and quantile mean targets, and
uses the positive part of the complete-window mean IC. Its graph identity
serializes both `window` and `quantiles`.

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
