"""Kelly-criterion-inspired transformers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .core import transformer


@transformer
def kelly(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """Estimate Kelly weights as rolling mean return divided by variance."""

    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("kelly window must be a positive integer")
    rolling = frame.rolling(window)
    mean = cast(pd.DataFrame, rolling.mean())
    variance = cast(pd.DataFrame, rolling.var())
    return mean.div(variance.replace(0, np.nan))


@transformer
def kelly_nonan_standardize(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """Fill missing values, standardize cross-sectionally, then estimate Kelly."""

    filled = frame.fillna(0)
    standardized = filled.sub(filled.mean(axis=1), axis=0).div(
        filled.std(axis=1).replace(0, np.nan), axis=0
    )
    return kelly.operation(standardized, window=window)


@transformer
def kelly_rank_boxcox(
    frame: pd.DataFrame,
    *,
    window: int,
    lambda_: float = 0,
) -> pd.DataFrame:
    """Rank to positive values, apply Box-Cox, then estimate Kelly."""

    ranked = frame.rank(axis=1, pct=True)
    transformed = np.log(ranked) if lambda_ == 0 else ranked.pow(lambda_).sub(1).div(lambda_)
    return kelly.operation(transformed, window=window)


@transformer
def kelly_rescaling_weight(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """Estimate Kelly weights and clip them to [0, 1]."""

    return kelly.operation(frame, window=window).clip(lower=0, upper=1)
