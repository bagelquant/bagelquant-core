"""Rolling multi-input composers."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real

import numpy as np
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


def _validate_non_negative_real(value: float, *, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative real number")


def _validate_positive_integer(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _rolling_regression(
    target: pl.DataFrame,
    factors: tuple[pl.DataFrame, ...],
    *,
    window: int,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> pl.DataFrame:
    _validate_window(window, window)
    if not factors:
        raise ValueError("rolling regression requires at least one factor")

    data = target.rename({VALUE: "target"})
    factor_columns: list[str] = []
    for index, factor in enumerate(factors):
        column = f"factor_{index}"
        factor_columns.append(column)
        data = data.join(
            factor.rename({VALUE: column}),
            on=[TIME, ASSET_ID],
            how="inner",
        )
    data = data.sort([ASSET_ID, TIME])

    rows: list[dict[str, object]] = []
    for group in data.partition_by(ASSET_ID):
        target_values = np.array(group["target"], dtype=float)
        features = np.column_stack(
            [np.array(group[column], dtype=float) for column in factor_columns]
        )
        for current, row in enumerate(group.iter_rows(named=True)):
            prediction: float | None = None
            if current >= window and np.isfinite(features[current]).all():
                train_y = target_values[current - window : current]
                train_x = features[current - window : current]
                valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
                if valid.any():
                    design = np.column_stack([np.ones(valid.sum()), train_x[valid]])
                    coefficients = fit(design, train_y[valid])
                    prediction = float(
                        np.r_[1.0, features[current]] @ coefficients
                    )
            rows.append(
                {
                    TIME: row[TIME],
                    ASSET_ID: row[ASSET_ID],
                    VALUE: prediction,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            TIME: data.schema[TIME],
            ASSET_ID: data.schema[ASSET_ID],
            VALUE: pl.Float64,
        },
    ).sort([TIME, ASSET_ID])


def _ols_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _ridge_fit(alpha: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        return (
            np.linalg.pinv(design.T @ design + alpha * penalty)
            @ design.T
            @ target
        )

    return fit


def _elastic_fit(
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
        coefficients = np.zeros(design.shape[1])
        coefficients[0] = target.mean()
        for _ in range(max_iter):
            previous = coefficients.copy()
            for index in range(design.shape[1]):
                residual = (
                    target
                    - design @ coefficients
                    + design[:, index] * coefficients[index]
                )
                numerator = design[:, index] @ residual
                if index == 0:
                    coefficients[index] = numerator / (
                        design[:, index] @ design[:, index]
                    )
                    continue
                denominator = (
                    design[:, index] @ design[:, index]
                    + alpha * (1.0 - l1_ratio)
                )
                shrinkage = alpha * l1_ratio
                coefficients[index] = (
                    np.sign(numerator)
                    * max(abs(numerator) - shrinkage, 0.0)
                    / denominator
                )
            if np.max(np.abs(coefficients - previous)) <= tolerance:
                break
        return coefficients

    return fit


@composer
def rolling_ols(
    target: pl.DataFrame, *factors: pl.DataFrame, window: int
) -> pl.DataFrame:
    return _rolling_regression(target, factors, window=window, fit=_ols_fit)


@composer
def rolling_ridge(
    target: pl.DataFrame,
    *factors: pl.DataFrame,
    window: int,
    alpha: float = 1.0,
) -> pl.DataFrame:
    _validate_non_negative_real(alpha, name="rolling_ridge alpha")
    return _rolling_regression(
        target, factors, window=window, fit=_ridge_fit(float(alpha))
    )


@composer
def rolling_elastic_net(
    target: pl.DataFrame,
    *factors: pl.DataFrame,
    window: int,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> pl.DataFrame:
    _validate_non_negative_real(alpha, name="rolling_elastic_net alpha")
    if (
        not isinstance(l1_ratio, Real)
        or isinstance(l1_ratio, bool)
        or not 0 <= l1_ratio <= 1
    ):
        raise ValueError(
            "rolling_elastic_net requires alpha >= 0 and l1_ratio in [0, 1]"
        )
    _validate_positive_integer(max_iter, name="rolling_elastic_net max_iter")
    _validate_non_negative_real(tolerance, name="rolling_elastic_net tolerance")
    return _rolling_regression(
        target,
        factors,
        window=window,
        fit=_elastic_fit(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            max_iter=max_iter,
            tolerance=float(tolerance),
        ),
    )


@composer
def rolling_lasso(
    target: pl.DataFrame,
    *factors: pl.DataFrame,
    window: int,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> pl.DataFrame:
    return rolling_elastic_net.operation(
        target,
        *factors,
        window=window,
        alpha=alpha,
        l1_ratio=1.0,
        max_iter=max_iter,
        tolerance=tolerance,
    )
