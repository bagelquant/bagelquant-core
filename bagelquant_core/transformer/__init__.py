"""
Unary graph transformations.

Built-in transformers are grouped by behavior while this module keeps the
public import surface compact.
"""

from .basic import abs_value, diff, identity, negate, pct_change
from .boxcox import boxcox
from .category import category_demean, category_mean, category_rank, category_zscore
from .core import TRANSFORMER_REGISTRY, TransformerFunction, transformer
from .general import (
    constant,
    date_age_constraint,
    delta,
    denoise,
    lag,
    negonly,
    nonnans,
    notnan,
    posonly,
    rate_of_change,
    remove_repeaded,
    remove_repeated,
    replace_inf,
)
from .kelly import (
    kelly,
    kelly_nonan_standardize,
    kelly_rank_boxcox,
    kelly_rescaling_weight,
)
from .logarithmic import inv_log_sqrt_rank, log, log1p, log_rank, signed_log1p
from .missing import bfill, ffill, fillna, fillna_zero
from .normalization import min_max_scale, net_scale, normalize, rank, winsorize, zscore
from .outlier import trim, trim_quantile, truncate
from .power import power, signed_power, sqrt
from .ranking import logrank, nrank, rankpct
from .replace import non_nan_to_one, non_nan_to_zero, replace_non_nan
from .rolling import (
    ewm_mean,
    ewm_std,
    ewm_var,
    rolling_ew_std,
    rolling_ewm,
    rolling_ewm_fw,
    rolling_kurt,
    rolling_max,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_percentile,
    rolling_rank,
    rolling_skew,
    rolling_std,
    rolling_sum,
    rolling_var,
    rolling_zscore,
)
from .sign import abs, sign
from .variance_stabilization import anscombe, freeman
from .translation import demean, translate_to_pos
from .trigonometric import arccos, arcsin, arctan, arctanh, cos, fisher, sin, trig

__all__ = [
    "TRANSFORMER_REGISTRY",
    "TransformerFunction",
    "abs",
    "abs_value",
    "anscombe",
    "arccos",
    "arcsin",
    "arctan",
    "arctanh",
    "bfill",
    "boxcox",
    "category_demean",
    "category_mean",
    "category_rank",
    "category_zscore",
    "constant",
    "cos",
    "date_age_constraint",
    "delta",
    "demean",
    "denoise",
    "diff",
    "ewm_mean",
    "ewm_std",
    "ewm_var",
    "ffill",
    "fillna",
    "fillna_zero",
    "fisher",
    "freeman",
    "identity",
    "inv_log_sqrt_rank",
    "kelly",
    "kelly_nonan_standardize",
    "kelly_rank_boxcox",
    "kelly_rescaling_weight",
    "lag",
    "log",
    "log1p",
    "log_rank",
    "logrank",
    "min_max_scale",
    "negate",
    "negonly",
    "net_scale",
    "nonnans",
    "non_nan_to_one",
    "non_nan_to_zero",
    "normalize",
    "notnan",
    "nrank",
    "pct_change",
    "posonly",
    "power",
    "rank",
    "rankpct",
    "rate_of_change",
    "remove_repeaded",
    "remove_repeated",
    "replace_non_nan",
    "replace_inf",
    "rolling_ew_std",
    "rolling_ewm",
    "rolling_ewm_fw",
    "rolling_kurt",
    "rolling_max",
    "rolling_mean",
    "rolling_median",
    "rolling_min",
    "rolling_percentile",
    "rolling_rank",
    "rolling_skew",
    "rolling_std",
    "rolling_sum",
    "rolling_var",
    "rolling_zscore",
    "sign",
    "sin",
    "signed_log1p",
    "signed_power",
    "sqrt",
    "translate_to_pos",
    "transformer",
    "trig",
    "trim",
    "trim_quantile",
    "truncate",
    "winsorize",
    "zscore",
]
