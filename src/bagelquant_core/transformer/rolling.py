"""Rolling time-series transformers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, map_groups_numpy, panel_like
from .core import transformer


def _validate_window(window: int, min_periods: int | None) -> int:
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("rolling window must be a positive integer")
    if min_periods is None:
        return window
    if (
        not isinstance(min_periods, int)
        or isinstance(min_periods, bool)
        or min_periods < 0
    ):
        raise ValueError("rolling min_periods must be a non-negative integer")
    if min_periods > window:
        raise ValueError("rolling min_periods must not exceed window")
    return min_periods


def _rolling_expr(frame: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return panel_like(frame.sort([ASSET_ID, TIME]), expr.over(ASSET_ID))


@transformer
def rolling_mean(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_mean(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_std(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None, ddof: int = 1
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_std(
            window, min_samples=_validate_window(window, min_periods), ddof=ddof
        ),
    )


@transformer
def rolling_min(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_min(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_max(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_max(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_sum(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_sum(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_var(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None, ddof: int = 1
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_var(
            window, min_samples=_validate_window(window, min_periods), ddof=ddof
        ),
    )


@transformer
def rolling_median(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_expr(
        frame,
        pl.col(VALUE).rolling_median(
            window, min_samples=_validate_window(window, min_periods)
        ),
    )


@transformer
def rolling_skew(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return map_groups_numpy(
        frame, lambda values: _rolling_apply(values, window, minp, _skew)
    )


@transformer
def rolling_kurt(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return map_groups_numpy(
        frame, lambda values: _rolling_apply(values, window, minp, _kurt)
    )


@transformer
def rolling_percentile(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return map_groups_numpy(
        frame, lambda values: _rolling_apply(values, window, minp, _last_percentile)
    )


@transformer
def rolling_rank(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    return map_groups_numpy(
        frame, lambda values: _rolling_apply(values, window, minp, _last_rank)
    )


@transformer
def rolling_zscore(
    frame: pl.DataFrame, *, window: int, min_periods: int | None = None, ddof: int = 1
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)

    def calculate(values: np.ndarray) -> np.ndarray:
        return _rolling_apply(
            values, window, minp, lambda sample: _zscore_last(sample, ddof)
        )

    return map_groups_numpy(frame, calculate)


def _alpha(
    *,
    com: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float:
    values = [value is not None for value in (com, span, halflife, alpha)]
    if sum(values) != 1:
        raise ValueError("ewm requires exactly one of com, span, halflife, or alpha")
    if alpha is not None:
        return float(alpha)
    if com is not None:
        return 1.0 / (1.0 + float(com))
    if span is not None:
        return 2.0 / (float(span) + 1.0)
    return 1.0 - float(np.exp(np.log(0.5) / float(halflife)))


def _ewm_values(values: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    output = np.full(len(values), np.nan)
    current = np.nan
    seen = 0
    for index, value in enumerate(values):
        if np.isnan(value):
            output[index] = current if seen >= min_periods else np.nan
            continue
        seen += 1
        current = (
            value if np.isnan(current) else alpha * value + (1.0 - alpha) * current
        )
        output[index] = current if seen >= min_periods else np.nan
    return output


@transformer
def ewm_mean(
    frame: pl.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
) -> pl.DataFrame:
    del adjust, ignore_na
    resolved = _alpha(com=com, span=span, halflife=halflife, alpha=alpha)
    return map_groups_numpy(
        frame, lambda values: _ewm_values(values, resolved, min_periods)
    )


@transformer
def ewm_var(
    frame: pl.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    bias: bool = False,
) -> pl.DataFrame:
    del adjust, ignore_na
    resolved = _alpha(com=com, span=span, halflife=halflife, alpha=alpha)

    def calculate(values: np.ndarray) -> np.ndarray:
        mean = _ewm_values(values, resolved, 0)
        variance = _ewm_values((values - mean) ** 2, resolved, min_periods)
        return variance if bias else variance * (len(values) / max(len(values) - 1, 1))

    return map_groups_numpy(frame, calculate)


@transformer
def ewm_std(
    frame: pl.DataFrame,
    *,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    bias: bool = False,
) -> pl.DataFrame:
    variance = ewm_var.operation(
        frame,
        com=com,
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
        bias=bias,
    )
    return panel_like(variance, pl.col(VALUE).sqrt())


rolling_ewm = ewm_mean
rolling_ew_std = ewm_std


@transformer
def rolling_ewm_fw(
    frame: pl.DataFrame, *, halflife: float, min_periods: int = 0
) -> pl.DataFrame:
    if halflife <= 0:
        raise ValueError("rolling_ewm_fw halflife must be positive")
    return ewm_mean.operation(frame, halflife=halflife, min_periods=min_periods)


def _rolling_apply(
    values: np.ndarray,
    window: int,
    min_periods: int,
    func: Callable[[np.ndarray], float],
) -> np.ndarray:
    output = np.full(len(values), np.nan)
    for index in range(len(values)):
        sample = values[max(0, index - window + 1) : index + 1]
        sample = sample[~np.isnan(sample)]
        if len(sample) >= min_periods:
            output[index] = func(sample)
    return output


def _skew(sample: np.ndarray) -> float:
    std = sample.std(ddof=1)
    return (
        np.nan
        if std == 0 or len(sample) < 3
        else float(np.mean(((sample - sample.mean()) / std) ** 3))
    )


def _kurt(sample: np.ndarray) -> float:
    std = sample.std(ddof=1)
    return (
        np.nan
        if std == 0 or len(sample) < 4
        else float(np.mean(((sample - sample.mean()) / std) ** 4) - 3.0)
    )


def _last_rank(sample: np.ndarray) -> float:
    return float(np.sum(sample <= sample[-1]))


def _last_percentile(sample: np.ndarray) -> float:
    return _last_rank(sample) / len(sample)


def _zscore_last(sample: np.ndarray, ddof: int) -> float:
    std = sample.std(ddof=ddof)
    return np.nan if std == 0 else float((sample[-1] - sample.mean()) / std)
