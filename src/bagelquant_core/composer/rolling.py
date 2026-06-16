"""Rolling multi-input composers."""

from __future__ import annotations

import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, panel_like
from .core import composer


def _validate_window(window: int, min_periods: int | None) -> int:
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("window must be positive")
    resolved = window if min_periods is None else min_periods
    if (
        not isinstance(resolved, int)
        or isinstance(resolved, bool)
        or resolved < 0
        or resolved > window
    ):
        raise ValueError("min_periods must be between 0 and window")
    return resolved


def _joined_pair(lhs: pl.DataFrame, rhs: pl.DataFrame) -> pl.DataFrame:
    return (
        lhs.rename({VALUE: "lhs"})
        .join(rhs.rename({VALUE: "rhs"}), on=[TIME, ASSET_ID], how="inner")
        .sort([ASSET_ID, TIME])
    )


def _valid_pair(data: pl.DataFrame) -> pl.DataFrame:
    valid = (
        pl.col("lhs").is_not_null()
        & pl.col("rhs").is_not_null()
        & ~pl.col("lhs").is_nan()
        & ~pl.col("rhs").is_nan()
    )
    return data.with_columns(
        pl.when(valid).then(pl.col("lhs")).otherwise(None).alias("y"),
        pl.when(valid).then(pl.col("rhs")).otherwise(None).alias("x"),
    )


@composer
def rolling_corr(
    lhs: pl.DataFrame, rhs: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    data = _joined_pair(lhs, rhs)
    return panel_like(
        data,
        pl.rolling_corr(
            pl.col("lhs").fill_nan(None),
            pl.col("rhs").fill_nan(None),
            window_size=window,
            min_samples=minp,
        ).over(ASSET_ID),
    )


@composer
def rolling_cov(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pl.DataFrame:
    minp = _validate_window(window, min_periods)
    data = _joined_pair(lhs, rhs)
    return panel_like(
        data,
        pl.rolling_cov(
            pl.col("lhs").fill_nan(None),
            pl.col("rhs").fill_nan(None),
            window_size=window,
            min_samples=minp,
            ddof=ddof,
        ).over(ASSET_ID),
    )


def _rolling_regression(
    target: pl.DataFrame, factor: pl.DataFrame, *, window: int, alpha: float = 0.0
) -> pl.DataFrame:
    _validate_window(window, window)
    data = _valid_pair(_joined_pair(target, factor))
    n = (
        pl.col("x")
        .is_not_null()
        .cast(pl.Float64)
        .rolling_sum(window, min_samples=1)
        .over(ASSET_ID)
    )
    sum_x = pl.col("x").rolling_sum(window, min_samples=window).over(ASSET_ID)
    sum_y = pl.col("y").rolling_sum(window, min_samples=window).over(ASSET_ID)
    sum_xx = (pl.col("x") * pl.col("x")).rolling_sum(
        window, min_samples=window
    ).over(ASSET_ID)
    sum_xy = (pl.col("x") * pl.col("y")).rolling_sum(
        window, min_samples=window
    ).over(ASSET_ID)
    centered_xx = sum_xx - (sum_x * sum_x) / n
    centered_xy = sum_xy - (sum_x * sum_y) / n
    slope = centered_xy / (centered_xx + float(alpha))
    return panel_like(data, pl.when(n >= window).then(slope).otherwise(None))


@composer
def rolling_ols(
    target: pl.DataFrame, factor: pl.DataFrame, *, window: int
) -> pl.DataFrame:
    return _rolling_regression(target, factor, window=window)


@composer
def rolling_ridge(
    target: pl.DataFrame, factor: pl.DataFrame, *, window: int, alpha: float = 1.0
) -> pl.DataFrame:
    return _rolling_regression(target, factor, window=window, alpha=alpha)


@composer
def rolling_elastic_net(
    target: pl.DataFrame,
    factor: pl.DataFrame,
    *,
    window: int,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> pl.DataFrame:
    del l1_ratio
    return _rolling_regression(target, factor, window=window, alpha=alpha)


@composer
def rolling_lasso(
    target: pl.DataFrame, factor: pl.DataFrame, *, window: int, alpha: float = 1.0
) -> pl.DataFrame:
    return _rolling_regression(target, factor, window=window, alpha=alpha)
