"""
Multi-input graph compositions.

Built-in composers are grouped by behavior while this module keeps the public
import surface compact.
"""

from .aggregation import (
    maximum,
    mean,
    minimum,
    product,
    sum_frames,
    weighted_mean,
    weighted_sum,
)
from .arithmetic import add, div, mul, sub
from .core import COMPOSER_REGISTRY, ComposerFunction, composer
from .general import coalesce
from .math import (
    and_,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    or_,
    power,
    power_df,
    xand,
    xor,
)
from .rolling import (
    rolling_corr,
    rolling_cov,
)

__all__ = [
    "COMPOSER_REGISTRY",
    "ComposerFunction",
    "add",
    "and_",
    "coalesce",
    "composer",
    "div",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "maximum",
    "mean",
    "minimum",
    "mul",
    "or_",
    "power",
    "power_df",
    "product",
    "rolling_corr",
    "rolling_cov",
    "sub",
    "sum_frames",
    "weighted_mean",
    "weighted_sum",
    "xand",
    "xor",
]
