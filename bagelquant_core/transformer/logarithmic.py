"""Logarithmic transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def log(frame: pd.DataFrame) -> pd.DataFrame:
    """Return natural logarithms, using NaN for non-positive values."""

    return np.log(frame.where(frame > 0))


@transformer
def log1p(frame: pd.DataFrame) -> pd.DataFrame:
    """Return log(1 + value), using NaN for values at or below -1."""

    return np.log1p(frame.where(frame > -1))


@transformer
def signed_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sign(value) * log(1 + abs(value))."""

    return np.sign(frame) * np.log1p(frame.abs())
