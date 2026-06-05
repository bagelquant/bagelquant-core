"""Cross-sectional outlier handling transformers."""

from __future__ import annotations

from numbers import Real

import pandas as pd

from .core import transformer


@transformer
def truncate(frame: pd.DataFrame, *, lower: Real, upper: Real) -> pd.DataFrame:
    """Clip values to fixed lower and upper bounds."""

    if lower > upper:
        raise ValueError("truncate requires lower <= upper")
    return frame.clip(lower=lower, upper=upper)


@transformer
def trim(frame: pd.DataFrame, *, lower: Real, upper: Real) -> pd.DataFrame:
    """Replace values outside fixed lower and upper bounds with NaN."""

    if lower > upper:
        raise ValueError("trim requires lower <= upper")
    return frame.where(frame.ge(lower) & frame.le(upper))


@transformer
def trim_quantile(
    frame: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Replace row values outside cross-sectional quantile bounds with NaN."""

    if not 0 <= lower <= upper <= 1:
        raise ValueError("trim_quantile requires 0 <= lower <= upper <= 1")
    lower_bound = frame.quantile(lower, axis=1)
    upper_bound = frame.quantile(upper, axis=1)
    return frame.where(
        frame.ge(lower_bound, axis=0) & frame.le(upper_bound, axis=0)
    )
