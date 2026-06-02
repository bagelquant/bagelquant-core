"""Cross-sectional normalization transformers."""

from __future__ import annotations

import pandas as pd

from .core import transformer


@transformer
def rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return percentile ranks across assets for each row."""

    return frame.rank(axis=1, pct=True)


@transformer
def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Return z-scores across assets for each row."""

    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, float("nan"))
    return frame.sub(mean, axis=0).div(std, axis=0)


@transformer
def winsorize(
    frame: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip each row to its lower and upper quantiles."""

    if not 0 <= lower <= upper <= 1:
        raise ValueError("winsorize requires 0 <= lower <= upper <= 1")
    lower_bound = frame.quantile(lower, axis=1)
    upper_bound = frame.quantile(upper, axis=1)
    return frame.clip(lower=lower_bound, upper=upper_bound, axis=0)


@transformer
def min_max_scale(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale each row to [0, 1], using NaN for constant rows."""

    minimum = frame.min(axis=1)
    spread = frame.max(axis=1).sub(minimum).replace(0, float("nan"))
    return frame.sub(minimum, axis=0).div(spread, axis=0)


@transformer
def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale each row linearly to [-1, 1]."""

    return min_max_scale.operation(frame).mul(2).sub(1)


@transformer
def net_scale(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale positive and negative row values independently by their sums."""

    positive = frame.clip(lower=0)
    negative = frame.clip(upper=0)
    positive_sum = positive.sum(axis=1).replace(0, float("nan"))
    negative_sum = negative.abs().sum(axis=1).replace(0, float("nan"))
    output = positive.div(positive_sum, axis=0).fillna(0) + negative.div(
        negative_sum, axis=0
    ).fillna(0)
    return output.where(frame.notna())
