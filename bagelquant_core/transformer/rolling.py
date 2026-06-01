"""Rolling time-series transformers."""

from __future__ import annotations

import pandas as pd

from .core import transformer


def _rolling(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None,
) -> "pd.core.window.rolling.Rolling":
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("rolling window must be a positive integer")
    if min_periods is not None and (
        not isinstance(min_periods, int)
        or isinstance(min_periods, bool)
        or min_periods < 0
    ):
        raise ValueError("rolling min_periods must be a non-negative integer")
    if min_periods is not None and min_periods > window:
        raise ValueError("rolling min_periods must not exceed window")
    return frame.rolling(window=window, min_periods=min_periods)


@transformer
def rolling_mean(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling means over time."""

    return _rolling(frame, window=window, min_periods=min_periods).mean()


@transformer
def rolling_std(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Return rolling standard deviations over time."""

    return _rolling(frame, window=window, min_periods=min_periods).std(ddof=ddof)


@transformer
def rolling_min(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling minimum values over time."""

    return _rolling(frame, window=window, min_periods=min_periods).min()


@transformer
def rolling_max(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling maximum values over time."""

    return _rolling(frame, window=window, min_periods=min_periods).max()


@transformer
def rolling_sum(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling sums over time."""

    return _rolling(frame, window=window, min_periods=min_periods).sum()
