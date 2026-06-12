# Transformer

## Overview

A transformer is a unary function-style operation:

```text
Panel | Graph -> Graph
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
| Basic | `identity`, `abs_value`, `negate`, `diff`, `pct_change` |
| Missing values | `fillna`, `fillna_zero`, `ffill`, `bfill` |
| Replacement | `replace_non_nan`, `non_nan_to_one`, `non_nan_to_zero` |
| Rolling | `rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`, `rolling_sum`, `ewm_mean`, `ewm_std`, `ewm_var` |
| Power | `power`, `signed_power`, `sqrt` |
| Logarithmic | `log`, `log1p`, `signed_log1p` |
| Normalization | `rank`, `zscore`, `winsorize`, `min_max_scale` |
| Category | `category_demean`, `category_mean`, `category_rank`, `category_zscore` |
| General | `nonnans`, `notnan`, `denoise`, `posonly`, `negonly`, `lag`, `delta`, `rate_of_change`, `remove_repeated`, `date_age_constraint`, `constant`, `replace_inf` |
| Translation | `demean`, `translate_to_pos` |
| Rank | `rankpct`, `nrank`, `logrank` |
| Outliers | `truncate`, `trim`, `trim_quantile` |
| Variance stabilization | `boxcox`, `anscombe`, `freeman`, `fisher` |
| Trigonometric | `sin`, `cos`, `arcsin`, `arccos`, `trig`, `arctanh`, `arctan` |
| Kelly criterion | `kelly`, `kelly_nonan_standardize`, `kelly_rank_boxcox`, `kelly_rescaling_weight` |

## Basic

Basic operations are element-wise or run over rows, which represent time:

| Transformer | Behavior |
| --- | --- |
| `identity(source)` | Return input values unchanged. |
| `abs_value(source)` | Return absolute values. |
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

Rolling operations run over rows, which represent time:

| Transformer | Behavior |
| --- | --- |
| `rolling_mean(source, window, min_periods=None)` | Rolling arithmetic mean. |
| `rolling_std(source, window, min_periods=None, ddof=1)` | Rolling standard deviation. |
| `rolling_min(source, window, min_periods=None)` | Rolling minimum. |
| `rolling_max(source, window, min_periods=None)` | Rolling maximum. |
| `rolling_sum(source, window, min_periods=None)` | Rolling sum. |
| `ewm_mean(source, ...)` | Pandas exponentially weighted mean. |
| `ewm_std(source, ...)` | Pandas exponentially weighted standard deviation. |
| `ewm_var(source, ...)` | Pandas exponentially weighted variance. |

EWM operations follow pandas semantics and require exactly one decay argument:
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

Normalization operations run across columns, which represent assets:

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
`rolling_ewm` and `rolling_ew_std` are half-life-compatible aliases for the
general EWM operations, while `rolling_ewm_fw` exposes expanding exponentially
weighted means.

## Category

Category operations accept a numeric source and a matching `CategoryPanel`.
The category panel may contain strings such as industry, sector, or country
labels:

```python
import pandas as pd

from bagelquant_core import CategoryPanel
from bagelquant_core.transformer import category_demean, category_rank

industry = CategoryPanel.from_domain(
    pd.DataFrame(...),
    domain,
    name="industry",
)

industry_neutral = category_demean(raw_factor, industry)
industry_ranked = category_rank(raw_factor, industry)
```

| Transformer | Behavior |
| --- | --- |
| `category_demean(source, categories)` | Subtract each category mean within each row. |
| `category_mean(source, categories)` | Replace values with their category mean within each row. |
| `category_rank(source, categories)` | Calculate percentile ranks within each category and row. |
| `category_zscore(source, categories)` | Calculate z-scores within each category and row. |

Although category operations are exported with transformers, they consume two
aligned inputs and are represented internally as multi-input graph nodes.

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
