"""Rolling time-series transformers."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from .core import transformer


def _rolling(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None,
) -> Any:
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


def _validate_min_periods(min_periods: int) -> None:
    if not isinstance(min_periods, int) or isinstance(min_periods, bool):
        raise TypeError("ewm min_periods must be an integer")
    if min_periods < 0:
        raise ValueError("ewm min_periods must be non-negative")


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


@transformer
def rolling_var(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Return rolling variances over time."""

    return _rolling(frame, window=window, min_periods=min_periods).var(ddof=ddof)


@transformer
def rolling_skew(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling skewness over time."""

    return _rolling(frame, window=window, min_periods=min_periods).skew()


@transformer
def rolling_kurt(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling kurtosis over time."""

    return _rolling(frame, window=window, min_periods=min_periods).kurt()


@transformer
def rolling_median(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling medians over time."""

    return _rolling(frame, window=window, min_periods=min_periods).median()


@transformer
def rolling_percentile(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return each value's percentile rank within its trailing window."""

    return _rolling(frame, window=window, min_periods=min_periods).rank(pct=True)


@transformer
def rolling_rank(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return each value's rank within its trailing window."""

    return _rolling(frame, window=window, min_periods=min_periods).rank()


@transformer
def rolling_zscore(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Return trailing-window z-scores for each current value."""

    rolling = _rolling(frame, window=window, min_periods=min_periods)
    return frame.sub(rolling.mean()).div(rolling.std(ddof=ddof).replace(0, float("nan")))


def _ewm(
    frame: pd.DataFrame,
    *,
    com: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
    min_periods: int,
    adjust: bool,
    ignore_na: bool,
) -> Any:
    decay_arguments = (com, span, halflife, alpha)
    if sum(value is not None for value in decay_arguments) != 1:
        raise ValueError("ewm requires exactly one of com, span, halflife, or alpha")
    _validate_min_periods(min_periods)
    return frame.ewm(
        com=com,
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
    )


@transformer
def ewm_mean(
    frame: pd.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
) -> pd.DataFrame:
    """Return pandas exponentially weighted means over time."""

    return _ewm(
        frame,
        com=com,
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
    ).mean()


@transformer
def ewm_std(
    frame: pd.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    bias: bool = False,
) -> pd.DataFrame:
    """Return pandas exponentially weighted standard deviations over time."""

    return _ewm(
        frame,
        com=com,
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
    ).std(bias=bias)


@transformer
def ewm_var(
    frame: pd.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    bias: bool = False,
) -> pd.DataFrame:
    """Return pandas exponentially weighted variances over time."""

    return _ewm(
        frame,
        com=com,
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
    ).var(bias=bias)


rolling_ewm = ewm_mean
rolling_ew_std = ewm_std


@transformer
def rolling_ewm_fw(
    frame: pd.DataFrame,
    *,
    halflife: float,
    min_periods: int = 0,
) -> pd.DataFrame:
    """Return expanding exponentially weighted means with a half-life."""

    if halflife <= 0:
        raise ValueError("rolling_ewm_fw halflife must be positive")
    _validate_min_periods(min_periods)
    return cast(
        pd.DataFrame,
        frame.ewm(halflife=halflife, min_periods=min_periods, adjust=True).mean(),
    )
