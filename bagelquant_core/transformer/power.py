"""Power transformers."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from .core import transformer


def _validate_exponent(exponent: Real) -> None:
    if not isinstance(exponent, Real) or isinstance(exponent, bool):
        raise TypeError("power exponent must be a real number")


@transformer
def power(frame: pd.DataFrame, *, exponent: Real) -> pd.DataFrame:
    """Raise each value to an exponent."""

    _validate_exponent(exponent)
    return frame.pow(exponent)


@transformer
def signed_power(frame: pd.DataFrame, *, exponent: Real) -> pd.DataFrame:
    """Raise absolute values to an exponent while preserving signs."""

    _validate_exponent(exponent)
    return np.sign(frame) * frame.abs().pow(exponent)


@transformer
def sqrt(frame: pd.DataFrame) -> pd.DataFrame:
    """Return square roots, using NaN for negative values."""

    return frame.where(frame >= 0).pow(0.5)
