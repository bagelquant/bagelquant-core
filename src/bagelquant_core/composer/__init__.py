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
    "divide",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mul",
    "multiply",
    "or_",
    "power",
    "power_df",
    "product",
    "rolling_corr",
    "rolling_cov",
    "sub",
    "subtract",
    "sum_frames",
    "weighted_mean",
    "weighted_sum",
    "xand",
    "xor",
]
