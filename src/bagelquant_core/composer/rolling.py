"""Rolling multi-input time-series composers."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd

from .core import composer


def _validate_window(window: int) -> None:
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("rolling window must be a positive integer")


def _validate_min_periods(min_periods: int | None, *, window: int) -> None:
    if min_periods is None:
        return
    if not isinstance(min_periods, int) or isinstance(min_periods, bool):
        raise TypeError("rolling min_periods must be an integer")
    if min_periods < 0:
        raise ValueError("rolling min_periods must be non-negative")
    if min_periods > window:
        raise ValueError("rolling min_periods must not exceed window")


def _validate_non_negative_real(value: float, *, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative real number")


@composer
def rolling_corr(
    lhs: pd.DataFrame,
    rhs: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling correlations between corresponding columns."""

    _validate_window(window)
    _validate_min_periods(min_periods, window=window)
    return cast(pd.DataFrame, lhs.rolling(window, min_periods=min_periods).corr(rhs))


@composer
def rolling_cov(
    lhs: pd.DataFrame,
    rhs: pd.DataFrame,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Return rolling covariance between corresponding columns."""

    _validate_window(window)
    _validate_min_periods(min_periods, window=window)
    return cast(
        pd.DataFrame,
        lhs.rolling(window, min_periods=min_periods).cov(rhs, ddof=ddof),
    )


def _rolling_regression(
    y: pd.DataFrame,
    factors: tuple[pd.DataFrame, ...],
    *,
    window: int,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    _validate_window(window)
    if not factors:
        raise ValueError("rolling regression requires at least one factor")
    output = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    column_positions = {column: position for position, column in enumerate(y.columns)}
    for column, column_position in column_positions.items():
        target = y[column].to_numpy(dtype=float)
        features = np.column_stack(
            [factor[column].to_numpy(dtype=float) for factor in factors]
        )
        for current in range(window, len(y)):
            train_x = features[current - window : current]
            train_y = target[current - window : current]
            current_x = features[current]
            valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
            if valid.sum() == 0 or not np.isfinite(current_x).all():
                continue
            design = np.column_stack([np.ones(valid.sum()), train_x[valid]])
            coefficients = fit(design, train_y[valid])
            output.iat[current, column_position] = np.r_[1, current_x] @ coefficients
    return output


def _ols_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _ridge_fit(alpha: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0
        return np.linalg.pinv(design.T @ design + alpha * penalty) @ design.T @ target

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
                residual = target - design @ coefficients + design[:, index] * coefficients[index]
                numerator = design[:, index] @ residual
                if index == 0:
                    coefficients[index] = numerator / (design[:, index] @ design[:, index])
                    continue
                denominator = design[:, index] @ design[:, index] + alpha * (1 - l1_ratio)
                shrinkage = alpha * l1_ratio
                coefficients[index] = np.sign(numerator) * max(abs(numerator) - shrinkage, 0) / denominator
            if np.max(np.abs(coefficients - previous)) <= tolerance:
                break
        return coefficients

    return fit


@composer
def rolling_ols(
    y: pd.DataFrame,
    *factors: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Predict y from factors fitted on the prior rolling window."""

    return _rolling_regression(y, factors, window=window, fit=_ols_fit)


@composer
def rolling_ridge(
    y: pd.DataFrame,
    *factors: pd.DataFrame,
    window: int,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Return prior-window ridge-regression predictions."""

    _validate_non_negative_real(alpha, name="rolling_ridge alpha")
    return _rolling_regression(y, factors, window=window, fit=_ridge_fit(alpha))


@composer
def rolling_elastic_net(
    y: pd.DataFrame,
    *factors: pd.DataFrame,
    window: int,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return prior-window elastic-net predictions."""

    _validate_non_negative_real(alpha, name="rolling_elastic_net alpha")
    if not isinstance(l1_ratio, Real) or isinstance(l1_ratio, bool) or not 0 <= l1_ratio <= 1:
        raise ValueError("rolling_elastic_net requires alpha >= 0 and l1_ratio in [0, 1]")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("rolling_elastic_net max_iter must be a positive integer")
    _validate_non_negative_real(tolerance, name="rolling_elastic_net tolerance")
    return _rolling_regression(
        y,
        factors,
        window=window,
        fit=_elastic_fit(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tolerance=tolerance,
        ),
    )


@composer
def rolling_lasso(
    y: pd.DataFrame,
    *factors: pd.DataFrame,
    window: int,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return prior-window lasso-regression predictions."""

    return rolling_elastic_net.operation(
        y,
        *factors,
        window=window,
        alpha=alpha,
        l1_ratio=1,
        max_iter=max_iter,
        tolerance=tolerance,
    )
