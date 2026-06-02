"""
Unary graph transformations.

Built-in transformers are grouped by behavior while this module keeps the
public import surface compact.
"""

from .basic import abs_value, diff, identity, negate, pct_change
from .category import category_demean, category_mean, category_rank, category_zscore
from .core import TRANSFORMER_REGISTRY, TransformerFunction, transformer
from .logarithmic import log, log1p, signed_log1p
from .missing import bfill, ffill, fillna, fillna_zero
from .normalization import min_max_scale, rank, winsorize, zscore
from .power import power, signed_power, sqrt
from .replace import non_nan_to_one, non_nan_to_zero, replace_non_nan
from .rolling import (
    ewm_mean,
    ewm_std,
    ewm_var,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
)

__all__ = [
    "TRANSFORMER_REGISTRY",
    "TransformerFunction",
    "abs_value",
    "bfill",
    "category_demean",
    "category_mean",
    "category_rank",
    "category_zscore",
    "diff",
    "ewm_mean",
    "ewm_std",
    "ewm_var",
    "ffill",
    "fillna",
    "fillna_zero",
    "identity",
    "log",
    "log1p",
    "min_max_scale",
    "negate",
    "non_nan_to_one",
    "non_nan_to_zero",
    "pct_change",
    "power",
    "rank",
    "replace_non_nan",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_std",
    "rolling_sum",
    "signed_log1p",
    "signed_power",
    "sqrt",
    "transformer",
    "winsorize",
    "zscore",
]
