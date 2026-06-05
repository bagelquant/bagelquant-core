"""Basic element-wise and time-series transformers."""

from __future__ import annotations

import pandas as pd

from .core import transformer


@transformer
def identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the input values unchanged."""

    return frame.copy()


@transformer
def abs_value(frame: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise absolute values."""

    return frame.abs()


@transformer
def negate(frame: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise negated values."""

    return -frame


@transformer
def diff(frame: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Return changes over a number of rows."""

    if not isinstance(periods, int) or isinstance(periods, bool):
        raise TypeError("diff periods must be an integer")
    if periods == 0:
        raise ValueError("diff periods must not be zero")
    return frame.diff(periods=periods)


@transformer
def pct_change(frame: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Return fractional changes over a number of rows."""

    if not isinstance(periods, int) or isinstance(periods, bool):
        raise TypeError("pct_change periods must be an integer")
    if periods == 0:
        raise ValueError("pct_change periods must not be zero")
    return frame.pct_change(periods=periods)
