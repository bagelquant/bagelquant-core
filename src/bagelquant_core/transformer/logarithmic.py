"""Logarithmic transformers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def log(frame: pd.DataFrame) -> pd.DataFrame:
    """Return natural logarithms, using NaN for non-positive values."""

    return cast(pd.DataFrame, np.log(frame.where(frame > 0)))


@transformer
def log1p(frame: pd.DataFrame) -> pd.DataFrame:
    """Return log(1 + value), using NaN for values at or below -1."""

    return cast(pd.DataFrame, np.log1p(frame.where(frame > -1)))


@transformer
def signed_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sign(value) * log(1 + abs(value))."""

    return cast(pd.DataFrame, np.sign(frame) * np.log1p(frame.abs()))


@transformer
def log_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return logarithms of cross-sectional percentile ranks."""

    return cast(pd.DataFrame, np.log(frame.rank(axis=1, pct=True)))


@transformer
def inv_log_sqrt_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return -log(rank(value)) / sqrt(rank(value))."""

    ranked = frame.rank(axis=1, pct=True)
    return cast(pd.DataFrame, -np.log(ranked) / np.sqrt(ranked))
