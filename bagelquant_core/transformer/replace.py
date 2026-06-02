"""Value replacement transformers."""

from __future__ import annotations

from numbers import Real

import pandas as pd

from .core import transformer


@transformer
def replace_non_nan(frame: pd.DataFrame, *, value: Real) -> pd.DataFrame:
    """Replace existing non-missing values with a numeric scalar."""

    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("replacement value must be a real number")
    return frame.where(frame.isna(), value)


@transformer
def non_nan_to_one(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace existing non-missing values with one."""

    return frame.where(frame.isna(), 1)


@transformer
def non_nan_to_zero(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace existing non-missing values with zero."""

    return frame.where(frame.isna(), 0)
