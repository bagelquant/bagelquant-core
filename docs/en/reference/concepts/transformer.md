# Transformer

## Overview

A transformer has exactly one semantic input Panel and may declare named,
keyword-only auxiliary Panel parameters:

```text
source: Panel | Graph
panel_parameters: named Panel | Graph dependencies
-> Graph[Panel]
```

For signatures, parameter descriptions, and examples for every public
operation, see the [transformer reference](../transformers/index.md).

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
| Basic | `identity`, `abs`, `negate`, `diff`, `pct_change` |
| Missing values | `fillna`, `fillna_zero`, `ffill`, `bfill` |
| Replacement | `replace_non_nan`, `non_nan_to_one`, `non_nan_to_zero` |
| Rolling | `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, `rolling_sum`, `ewm_mean`, `ewm_std`, `ewm_var` |
| Power | `power`, `signed_power`, `sqrt` |
| Logarithmic | `log`, `log1p`, `signed_log1p` |
| Normalization | `rank`, `zscore`, `winsorize`, `min_max_scale` |
| Group & neutralization | `group_demean`, `group_mean`, `group_rank`, `group_percentile`, `group_zscore`, `orthogonalize` |
| Masking & scaling | `mask`, `project`, `vol_scale` |
| Rolling regression | `rolling_ols`, `rolling_ridge`, `rolling_lasso`, `rolling_elastic_net` |
| General | `notnan`, `denoise`, `posonly`, `negonly`, `lag`, `remove_repeated`, `date_age_constraint`, `constant`, `replace_inf` |
| Streaks | `repeat_count`, `diff_from_last_change`, `pct_change_from_last_change`, `streak_count` |
| Translation | `demean`, `translate_to_pos` |
| Rank | `rankpct`, `nrank`, `log_rank` |
| Outliers | `truncate`, `trim`, `trim_quantile` |
| Variance stabilization | `boxcox`, `anscombe`, `freeman`, `arctanh` |
| Trigonometric | `sin`, `cos`, `arcsin`, `arccos`, `trig`, `arctanh`, `arctan` |
| Kelly criterion | `kelly`, `kelly_nonan_standardize`, `kelly_rank_boxcox`, `kelly_rescaling_weight` |

## Basic

Basic operations are element-wise or run per `asset_id` ordered by `time`:

| Transformer | Behavior |
| --- | --- |
| `identity(source)` | Return input values unchanged. |
| `abs(source)` | Return absolute values. |
| `negate(source)` | Negate values. |
| `diff(source, periods=1)` | Calculate differences over time. |
| `pct_change(source, periods=1)` | Calculate fractional changes over time, such as returns from a price panel. |

## Missing Values

Missing-value operations preserve the panel shape:

| Transformer | Behavior |
| --- | --- |
| `fillna(source, value=0)` | Fill `NaN` values with a numeric scalar. |
| `fillna_zero(source)` | Fill `NaN` values with zero. |
| `ffill(source, limit=None)` | Forward-fill over time. |
| `bfill(source, limit=None)` | Backward-fill over time. |

`ffill` and `bfill` accept an optional positive `limit`.

## Replacement

Replacement operations preserve missing values and replace existing values:

| Transformer | Behavior |
| --- | --- |
| `replace_non_nan(source, value=...)` | Replace each non-`NaN` value with a numeric scalar. |
| `non_nan_to_one(source)` | Replace each non-`NaN` value with one. |
| `non_nan_to_zero(source)` | Replace each non-`NaN` value with zero. |

These operations are useful for availability masks and constant exposures.

## Rolling

Rolling operations run per `asset_id` ordered by `time`:

| Transformer | Behavior |
| --- | --- |
| `rolling_mean(source, window, min_periods=None)` | Rolling arithmetic mean. |
| `rolling_std(source, window, min_periods=None, ddof=1)` | Rolling standard deviation. |
| `rolling_min(source, window, min_periods=None)` | Rolling minimum. |
| `rolling_max(source, window, min_periods=None)` | Rolling maximum. |
| `rolling_sum(source, window, min_periods=None)` | Rolling sum. |
| `ewm_mean(source, ...)` | Exponentially weighted mean. |
| `ewm_std(source, ...)` | Exponentially weighted standard deviation. |
| `ewm_var(source, ...)` | Exponentially weighted variance. |

EWM operations require exactly one decay argument:
`com`, `span`, `halflife`, or `alpha`. They also accept `min_periods`,
`adjust`, and `ignore_na`. `ewm_std` and `ewm_var` additionally accept `bias`.

## Power

| Transformer | Behavior |
| --- | --- |
| `power(source, exponent)` | Raise values to an exponent. |
| `signed_power(source, exponent)` | Raise absolute values to an exponent while preserving signs. |
| `sqrt(source)` | Calculate square roots, returning `NaN` for negative values. |

## Logarithmic

| Transformer | Behavior |
| --- | --- |
| `log(source)` | Calculate natural logarithms, returning `NaN` for non-positive values. |
| `log1p(source)` | Calculate `log(1 + value)`, returning `NaN` for values at or below `-1`. |
| `signed_log1p(source)` | Calculate `sign(value) * log(1 + abs(value))`. |

## Normalization

Normalization operations run cross-sectionally by `time`:

| Transformer | Behavior |
| --- | --- |
| `rank(source)` | Calculate percentile ranks for each row. |
| `zscore(source)` | Calculate z-scores for each row. |
| `winsorize(source, lower=0.01, upper=0.99)` | Clip each row to its quantile bounds. |
| `min_max_scale(source)` | Scale each row to `[0, 1]`. |
| `normalize(source)` | Scale each row to `[-1, 1]`. |
| `net_scale(source)` | Scale positive and negative values independently by their row sums. |

Constant rows produce `NaN` values where normalization is undefined.

## Extended Rolling Operations

The rolling family also includes `rolling_var`, `rolling_skew`, `rolling_kurt`,
`rolling_median`, `rolling_rank`, `rolling_percentile`, and `rolling_zscore`.
`rolling_ewm_fw` exposes expanding exponentially weighted means.

## Named Panel parameters

Auxiliary Panels are parameters rather than peer inputs. They still participate
in topology, Domain alignment, cache identity, and availability tracing. The
output availability time is the maximum availability across the source and all
auxiliary Panels.

Group operations accept a numeric source and a matching `CategoryPanel` through
the keyword-only `group` parameter. The category panel may contain strings such
as industry, sector, or country labels:

```python
import polars as pl

from bagelquant_core import CategoryPanel
from bagelquant_core.transformer import group_demean, group_percentile

industry = CategoryPanel.from_domain(
    pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02"],
            "asset_id": ["AAPL", "MSFT"],
            "value": ["tech", "software"],
        }
    ),
    domain,
    name="industry",
)

industry_neutral = group_demean(raw_factor, group=industry)
industry_percentile = group_percentile(raw_factor, group=industry)
```

| Transformer | Behavior |
| --- | --- |
| `group_demean(source, *, group=...)` | Subtract each group mean within each date. |
| `group_mean(source, *, group=...)` | Replace values with their group mean within each date. |
| `group_rank(source, *, group=...)` | Calculate ordinal ranks within each group and date. |
| `group_percentile(source, *, group=...)` | Calculate average percentile ranks within each group and date. |
| `group_zscore(source, *, group=...)` | Calculate z-scores within each group and date. |
| `orthogonalize(source, *, factors=(...), fit_intercept=False)` | Return same-date OLS residuals against one or more factor Panels. |

Other named Panel parameters include `mask_frame`, `binary`, `volatility`, and
the `factors` tuple used by rolling regressions. Each reference page includes
the actual source, Panel-parameter, and output tables produced by executing its
fixture.

## User-Defined Transformers

```python
import polars as pl

from bagelquant_core import (
    ExecutionMode,
    InputDensity,
    OperationContract,
    TraceRule,
)
from bagelquant_core.transformer import transformer


@transformer(
    contract=OperationContract(
        execution=ExecutionMode.LAZY,
        density=InputDensity.SPARSE_OK,
        trace_rule=TraceRule.PASSTHROUGH,
    )
)
def demean(frame: pl.LazyFrame) -> pl.LazyFrame:
    means = frame.group_by("time").agg(pl.col("value").mean().alias("mean"))
    return (
        frame.join(means, on="time")
        .with_columns((pl.col("value") - pl.col("mean")).alias("value"))
        .select("time", "asset_id", "value")
    )


centered = demean(price, name="centered")
```

The decorated function accepts a `Panel` or `Graph` while constructing a
workflow. A declared lazy operation receives a `LazyFrame`. A bare custom
decorator is intentionally conservative: it receives a dense `DataFrame` at
an eager barrier. If traced inputs are possible, declare a trace rule;
otherwise execution raises an explicit error.

Configuration arguments are stored in graph specifications and cache keys.
