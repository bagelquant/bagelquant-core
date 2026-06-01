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

__all__ = [
    "COMPOSER_REGISTRY",
    "ComposerFunction",
    "add",
    "composer",
    "div",
    "maximum",
    "mean",
    "minimum",
    "mul",
    "product",
    "sub",
    "sum_frames",
    "weighted_mean",
    "weighted_sum",
]
