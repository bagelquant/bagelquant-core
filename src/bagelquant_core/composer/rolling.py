"""Rolling multi-input composers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from ..frame import ASSET_ID, TIME, VALUE
from .core import composer


def _rolling_pair(
    lhs: pl.DataFrame,
    rhs: pl.DataFrame,
    *,
    window: int,
    min_periods: int | None,
    func: Callable[[np.ndarray, np.ndarray], float],
) -> pl.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")
    minp = window if min_periods is None else min_periods
    data = (
        lhs.rename({VALUE: "lhs"})
        .join(
            rhs.rename({VALUE: "rhs"}),
            on=[TIME, ASSET_ID],
            how="inner",
        )
        .sort([ASSET_ID, TIME])
    )
    rows: list[dict[str, object]] = []
    for group in data.partition_by(ASSET_ID):
        left = np.array(group["lhs"], dtype=float)
        right = np.array(group["rhs"], dtype=float)
        for index, row in enumerate(group.iter_rows(named=True)):
            start = max(0, index - window + 1)
            x = left[start : index + 1]
            y = right[start : index + 1]
            valid = ~(np.isnan(x) | np.isnan(y))
            value = func(x[valid], y[valid]) if valid.sum() >= minp else np.nan
            rows.append({TIME: row[TIME], ASSET_ID: row[ASSET_ID], VALUE: value})
    return pl.DataFrame(rows).sort([TIME, ASSET_ID])


@composer
def rolling_corr(
    lhs: pl.DataFrame, rhs: pl.DataFrame, *, window: int, min_periods: int | None = None
) -> pl.DataFrame:
    return _rolling_pair(
        lhs,
        rhs,
        window=window,
        min_periods=min_periods,
        func=lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan,
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
    return _rolling_pair(
        lhs,
        rhs,
        window=window,
        min_periods=min_periods,
        func=lambda x, y: (
            float(np.cov(x, y, ddof=ddof)[0, 1]) if len(x) > ddof else np.nan
        ),
    )


def _rolling_regression(
    target: pl.DataFrame, factor: pl.DataFrame, *, window: int, alpha: float = 0.0
) -> pl.DataFrame:
    return _rolling_pair(
        target,
        factor,
        window=window,
        min_periods=window,
        func=lambda y, x: _slope(y, x, alpha=alpha),
    )


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


def _slope(y: np.ndarray, x: np.ndarray, *, alpha: float) -> float:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0, alpha])
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return float(beta[1])
