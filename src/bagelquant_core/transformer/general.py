"""General-purpose transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def nonnans(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values with zero."""

    return frame.fillna(0)


@transformer
def notnan(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one where values are present and zero where they are missing."""

    return frame.notna().astype(float)


@transformer
def denoise(frame: pd.DataFrame, *, threshold: float = 1e-12) -> pd.DataFrame:
    """Replace values whose absolute magnitude is tiny with zero."""

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("denoise threshold must be a non-negative real number")
    return frame.mask(frame.abs() < threshold, 0)


@transformer
def posonly(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep positive values and zero, replacing negative values with NaN."""

    return frame.where(frame >= 0)


@transformer
def negonly(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep negative values and zero, replacing positive values with NaN."""

    return frame.where(frame <= 0)


@transformer
def lag(frame: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Shift values over rows, which represent time."""

    return frame.shift(periods=_validate_periods(periods, operation="lag"))


@transformer
def delta(frame: pd.DataFrame, *, interval: int = 1) -> pd.DataFrame:
    """Return changes between rows separated by an interval."""

    return frame.diff(periods=_validate_periods(interval, operation="delta"))


@transformer
def rate_of_change(frame: pd.DataFrame, *, interval: int = 1) -> pd.DataFrame:
    """Return row differences divided by the interval."""

    checked = _validate_periods(interval, operation="rate_of_change")
    return frame.diff(periods=checked).div(checked)


@transformer
def remove_repeated(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep values only when they changed from the previous row."""

    return frame.where(frame.ne(frame.shift()))


@transformer
def date_age_constraint(
    frame: pd.DataFrame,
    *,
    window: int,
    min_valid: int | None = None,
) -> pd.DataFrame:
    """Mask values until enough valid observations exist in a trailing window."""

    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("date_age_constraint window must be a positive integer")
    required = window if min_valid is None else min_valid
    if (
        not isinstance(required, int)
        or isinstance(required, bool)
        or required <= 0
        or required > window
    ):
        raise ValueError("date_age_constraint min_valid must be in [1, window]")
    return frame.where(frame.notna().rolling(window).sum() >= required)


@transformer
def constant(frame: pd.DataFrame, *, value: float = 1) -> pd.DataFrame:
    """Return a same-shaped constant frame."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("constant value must be a real number")
    return pd.DataFrame(value, index=frame.index, columns=frame.columns)


@transformer
def replace_inf(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace positive and negative infinity with NaN."""

    return frame.replace([np.inf, -np.inf], np.nan)


def _validate_periods(periods: int, *, operation: str) -> int:
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise TypeError(f"{operation} periods must be an integer")
    if periods == 0:
        raise ValueError(f"{operation} periods must not be zero")
    return periods
