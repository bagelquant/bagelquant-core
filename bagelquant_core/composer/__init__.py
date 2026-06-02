"""
Multi-input graph compositions.

Built-in composers are grouped by behavior while this module keeps the public
import surface compact.
"""

from .aggregation import (
    max,
    maximum,
    mean,
    min,
    minimum,
    product,
    sum_frames,
    weighted_mean,
    weighted_sum,
)
from .arithmetic import add, div, divide, mul, multiply, sub, subtract
from .core import COMPOSER_REGISTRY, ComposerFunction, composer
from .general import coalesce, mask, project
from .math import (
    and_,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_,
    or_,
    power,
    power_df,
    xand,
    xor,
)
from .rolling import (
    rolling_corr,
    rolling_cov,
    rolling_elastic_net,
    rolling_lasso,
    rolling_ols,
    rolling_ridge,
)
from .scaling import vol_scale
from .xsectional import (
    group_demean,
    group_max,
    group_mean,
    group_median,
    group_min,
    group_percentile,
    group_rank,
    group_rankpct,
    group_std,
    group_zscore,
    orthogonalize,
)

__all__ = [
    "COMPOSER_REGISTRY",
    "ComposerFunction",
    "add",
    "and_",
    "coalesce",
    "composer",
    "div",
    "divide",
    "equal",
    "greater",
    "greater_equal",
    "group_demean",
    "group_max",
    "group_mean",
    "group_median",
    "group_min",
    "group_percentile",
    "group_rank",
    "group_rankpct",
    "group_std",
    "group_zscore",
    "less",
    "less_equal",
    "mask",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mul",
    "multiply",
    "not_",
    "or_",
    "orthogonalize",
    "power",
    "power_df",
    "product",
    "project",
    "rolling_corr",
    "rolling_cov",
    "rolling_elastic_net",
    "rolling_lasso",
    "rolling_ols",
    "rolling_ridge",
    "sub",
    "subtract",
    "sum_frames",
    "vol_scale",
    "weighted_mean",
    "weighted_sum",
    "xand",
    "xor",
]
