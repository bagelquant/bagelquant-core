"""Rolling multi-input composers."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real

import numpy as np
import polars as pl

from ..frame import (
    ASSET_ID,
    TIME,
    VALUE,
    _balanced_inner_join,
    panel_like,
)
from .core import composer

_MAX_WORKING_BYTES = 64 * 1024 * 1024
_SINGLE_FACTOR_OLS_ARRAYS = 16
_GRAM_CONDITION_LIMIT = 1e-3
_RIDGE_CONDITION_LIMIT = 1e-10


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


def _joined_regression_data(
    target: pl.DataFrame,
    factors: tuple[pl.DataFrame, ...],
) -> tuple[pl.DataFrame, list[str]]:
    if not factors:
        raise ValueError("rolling regression requires at least one factor")

    factor_columns = [f"factor_{index}" for index in range(len(factors))]
    frames = [target.rename({VALUE: "target"})]
    frames.extend(
        factor.rename({VALUE: column})
        for factor, column in zip(factors, factor_columns, strict=True)
    )
    return _balanced_inner_join(frames).sort([ASSET_ID, TIME]), factor_columns


def _rolling_ols(
    target: pl.DataFrame,
    factors: tuple[pl.DataFrame, ...],
    *,
    window: int,
) -> pl.DataFrame:
    """Predict with batched prior-window least squares per asset."""

    _validate_window(window, window)
    if not factors:
        raise ValueError("rolling regression requires at least one factor")

    data, factor_columns = _joined_regression_data(target, factors)

    if len(factor_columns) == 1:
        return _rolling_single_factor_ols(
            data,
            factor_column=factor_columns[0],
            window=window,
        )

    return _rolling_gram_regression(
        data,
        factor_columns=factor_columns,
        window=window,
        method="ols",
    )


def _rolling_gram_regression(
    data: pl.DataFrame,
    *,
    factor_columns: list[str],
    window: int,
    method: str,
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    max_iter: int = 0,
    tolerance: float = 0.0,
) -> pl.DataFrame:
    """Predict rolling regressions from bounded sufficient statistics."""

    coefficient_count = len(factor_columns) + 1
    arrays_per_row = (
        2 * coefficient_count * coefficient_count
        + 6 * coefficient_count
        + 8
    )
    rows_per_batch = max(
        1,
        _MAX_WORKING_BYTES
        // (np.dtype(np.float64).itemsize * arrays_per_row)
        - window,
    )

    target_values = data.get_column("target").to_numpy().astype(
        float, copy=False
    )
    features = np.column_stack(
        [
            data.get_column(column).to_numpy().astype(float, copy=False)
            for column in factor_columns
        ]
    )
    prediction = np.full(len(data), np.nan, dtype=float)
    lengths = (
        data.group_by(ASSET_ID, maintain_order=True)
        .len()
        .get_column("len")
        .to_numpy()
    )
    group_offset = 0
    for length in lengths:
        group_end = group_offset + int(length)
        group_y = target_values[group_offset:group_end]
        group_x = features[group_offset:group_end]
        prediction_count = int(length) - window

        for start in range(0, max(prediction_count, 0), rows_per_batch):
            end = min(start + rows_per_batch, prediction_count)
            batch_size = end - start
            segment_y = group_y[start : window + end]
            segment_x = group_x[start : window + end]
            valid = np.isfinite(segment_y) & np.isfinite(segment_x).all(axis=1)
            design = np.zeros((len(segment_y), coefficient_count), dtype=float)
            design[:, 0] = valid
            design[:, 1:] = np.where(valid[:, None], segment_x, 0.0)
            valid_y = np.where(valid, segment_y, 0.0)

            counts = _window_sum(valid.astype(np.int64), window, batch_size)
            gram = _window_sum(
                np.einsum("ni,nj->nij", design, design),
                window,
                batch_size,
            )
            rhs = _window_sum(
                design * valid_y[:, None],
                window,
                batch_size,
            )
            current_x = group_x[window + start : window + end]
            eligible = (counts > 0) & np.isfinite(current_x).all(axis=1)
            current_design = np.column_stack(
                [np.ones(batch_size), current_x]
            )

            if method == "ols":
                coefficients, fast = _solve_ols_from_gram(
                    gram,
                    rhs,
                    counts=counts,
                    eligible=eligible,
                )
                fallback = eligible & ~fast
            elif method == "ridge":
                coefficients, fast = _solve_ridge_from_gram(
                    gram,
                    rhs,
                    eligible=eligible,
                    alpha=alpha,
                )
                fallback = eligible & ~fast
            else:
                coefficients = _solve_elastic_from_gram(
                    gram,
                    rhs,
                    counts=counts,
                    eligible=eligible,
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    tolerance=tolerance,
                )
                fallback = np.zeros(batch_size, dtype=bool)

            for index in np.flatnonzero(fallback):
                window_start = start + index
                exact_y = group_y[window_start : window_start + window]
                exact_x = group_x[window_start : window_start + window]
                exact_valid = (
                    np.isfinite(exact_y)
                    & np.isfinite(exact_x).all(axis=1)
                )
                exact_design = np.column_stack(
                    [np.ones(exact_valid.sum()), exact_x[exact_valid]]
                )
                if method == "ols":
                    coefficients[index] = _ols_fit(
                        exact_design,
                        exact_y[exact_valid],
                    )
                else:
                    coefficients[index] = _ridge_fit(alpha)(
                        exact_design,
                        exact_y[exact_valid],
                    )

            values = np.einsum("ij,ij->i", current_design, coefficients)
            prediction_start = group_offset + window + start
            prediction_end = group_offset + window + end
            prediction[prediction_start:prediction_end] = np.where(
                eligible,
                values,
                np.nan,
            )
        group_offset = group_end

    if data.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}
        )
    return (
        data.select(TIME, ASSET_ID)
        .with_columns(
            pl.Series(VALUE, prediction, nan_to_null=True)
        )
        .sort([TIME, ASSET_ID])
    )


def _solve_ols_from_gram(
    gram: np.ndarray,
    rhs: np.ndarray,
    *,
    counts: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues)
    cutoff = (
        np.finfo(np.float64).eps
        * np.maximum(counts, gram.shape[-1])[:, None]
        * singular_values[:, -1:]
    )
    full_rank = (singular_values > cutoff).all(axis=1)
    well_conditioned = (
        singular_values[:, 0]
        > _GRAM_CONDITION_LIMIT * singular_values[:, -1]
    )
    fast = eligible & full_rank & well_conditioned
    inverse = np.divide(
        1.0,
        eigenvalues,
        out=np.zeros_like(eigenvalues),
        where=fast[:, None],
    )
    projected = np.einsum(
        "bij,bj->bi",
        eigenvectors.swapaxes(-2, -1),
        rhs,
    )
    coefficients = np.einsum(
        "bij,bj->bi",
        eigenvectors,
        inverse * projected,
    )
    return coefficients, fast


def _solve_ridge_from_gram(
    gram: np.ndarray,
    rhs: np.ndarray,
    *,
    eligible: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    penalized = gram.copy()
    diagonal = np.arange(1, gram.shape[-1])
    penalized[:, diagonal, diagonal] += alpha
    eigenvalues, eigenvectors = np.linalg.eigh(penalized)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    fast = (
        eligible
        & (eigenvalues[:, 0] > 0)
        & (
            eigenvalues[:, 0]
            > _RIDGE_CONDITION_LIMIT * eigenvalues[:, -1]
        )
    )
    inverse = np.divide(
        1.0,
        eigenvalues,
        out=np.zeros_like(eigenvalues),
        where=fast[:, None],
    )
    projected = np.einsum(
        "bij,bj->bi",
        eigenvectors.swapaxes(-2, -1),
        rhs,
    )
    coefficients = np.einsum(
        "bij,bj->bi",
        eigenvectors,
        inverse * projected,
    )
    return coefficients, fast


def _solve_elastic_from_gram(
    gram: np.ndarray,
    rhs: np.ndarray,
    *,
    counts: np.ndarray,
    eligible: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    coefficient_count = gram.shape[-1]
    coefficients = np.zeros((len(gram), coefficient_count), dtype=float)
    coefficients[:, 0] = np.divide(
        rhs[:, 0],
        counts,
        out=np.zeros(len(gram), dtype=float),
        where=counts > 0,
    )
    active = eligible.copy()
    shrinkage = alpha * l1_ratio
    ridge_penalty = alpha * (1.0 - l1_ratio)

    for _ in range(max_iter):
        previous = coefficients.copy()
        for index in range(coefficient_count):
            numerator = (
                rhs[:, index]
                - np.einsum(
                    "bi,bi->b",
                    gram[:, index, :],
                    coefficients,
                )
                + gram[:, index, index] * coefficients[:, index]
            )
            if index == 0:
                updated = np.divide(
                    numerator,
                    gram[:, index, index],
                    out=np.full(len(gram), np.nan),
                    where=gram[:, index, index] != 0,
                )
            else:
                denominator = gram[:, index, index] + ridge_penalty
                updated = np.divide(
                    np.sign(numerator)
                    * np.maximum(np.abs(numerator) - shrinkage, 0.0),
                    denominator,
                    out=np.full(len(gram), np.nan),
                    where=denominator != 0,
                )
            coefficients[:, index] = np.where(
                active,
                updated,
                coefficients[:, index],
            )
        converged = (
            np.max(np.abs(coefficients - previous), axis=1)
            <= tolerance
        )
        active &= ~converged
        if not active.any():
            break
    return coefficients


def _rolling_single_factor_ols(
    data: pl.DataFrame,
    *,
    factor_column: str,
    window: int,
) -> pl.DataFrame:
    """Predict single-factor OLS from prior-window sufficient statistics."""

    itemsize = np.dtype(np.float64).itemsize
    rows_per_batch = max(
        1,
        _MAX_WORKING_BYTES // (itemsize * _SINGLE_FACTOR_OLS_ARRAYS) - window,
    )
    epsilon = np.finfo(np.float64).eps

    target_values = data.get_column("target").to_numpy().astype(
        float, copy=False
    )
    factor_values = data.get_column(factor_column).to_numpy().astype(
        float, copy=False
    )
    prediction = np.full(len(data), np.nan, dtype=float)
    lengths = (
        data.group_by(ASSET_ID, maintain_order=True)
        .len()
        .get_column("len")
        .to_numpy()
    )
    group_offset = 0
    for length in lengths:
        group_end = group_offset + int(length)
        group_y = target_values[group_offset:group_end]
        group_x = factor_values[group_offset:group_end]
        prediction_count = int(length) - window

        for start in range(0, max(prediction_count, 0), rows_per_batch):
            end = min(start + rows_per_batch, prediction_count)
            batch_size = end - start
            train_y = group_y[start : window + end]
            train_x = group_x[start : window + end]
            valid = np.isfinite(train_y) & np.isfinite(train_x)
            valid_y = np.where(valid, train_y, 0.0)
            valid_count = valid.sum()
            factor_shift = (
                float(np.where(valid, train_x, 0.0).sum() / valid_count)
                if valid_count
                else 0.0
            )
            shifted_x = np.where(valid, train_x - factor_shift, 0.0)

            counts = _window_sum(valid.astype(np.int64), window, batch_size)
            sum_shifted_x = _window_sum(shifted_x, window, batch_size)
            sum_y = _window_sum(valid_y, window, batch_size)
            sum_shifted_xx = _window_sum(
                shifted_x * shifted_x,
                window,
                batch_size,
            )
            sum_shifted_xy = _window_sum(
                shifted_x * valid_y,
                window,
                batch_size,
            )
            current_x = group_x[window + start : window + end]
            eligible = (counts > 0) & np.isfinite(current_x)
            sum_x = sum_shifted_x + counts * factor_shift
            sum_xx = (
                sum_shifted_xx
                + 2.0 * factor_shift * sum_shifted_x
                + counts * factor_shift * factor_shift
            )

            gram = np.empty((batch_size, 2, 2), dtype=float)
            gram[:, 0, 0] = counts
            gram[:, 0, 1] = sum_x
            gram[:, 1, 0] = sum_x
            gram[:, 1, 1] = sum_xx
            eigenvalues = np.linalg.eigvalsh(gram)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            singular_values = np.sqrt(eigenvalues)
            cutoff = (
                epsilon
                * np.maximum(counts, 2)
                * singular_values[:, 1]
            )
            full_rank = singular_values[:, 0] > cutoff
            well_conditioned = (
                singular_values[:, 0]
                > np.sqrt(epsilon) * singular_values[:, 1]
            )

            safe_counts = np.maximum(counts, 1)
            mean_shifted_x = sum_shifted_x / safe_counts
            mean_y = sum_y / safe_counts
            centered_xx = (
                sum_shifted_xx - sum_shifted_x * mean_shifted_x
            )
            centered_xy = (
                sum_shifted_xy - sum_shifted_x * mean_y
            )
            cancellation_scale = np.maximum.reduce(
                [
                    np.abs(sum_shifted_xx),
                    np.abs(sum_shifted_x * mean_shifted_x),
                    np.ones(batch_size),
                ]
            )
            stable_centering = centered_xx > 64.0 * epsilon * cancellation_scale
            fast = (
                eligible
                & full_rank
                & well_conditioned
                & stable_centering
            )

            values = np.full(batch_size, np.nan, dtype=float)
            slope = np.divide(
                centered_xy,
                centered_xx,
                out=np.zeros(batch_size, dtype=float),
                where=fast,
            )
            values[fast] = (
                mean_y[fast]
                + slope[fast]
                * (
                    current_x[fast]
                    - factor_shift
                    - mean_shifted_x[fast]
                )
            )

            for index in np.flatnonzero(eligible & ~fast):
                window_start = start + index
                exact_y = group_y[window_start : window_start + window]
                exact_x = group_x[window_start : window_start + window]
                exact_valid = np.isfinite(exact_y) & np.isfinite(exact_x)
                design = np.column_stack(
                    [np.ones(exact_valid.sum()), exact_x[exact_valid]]
                )
                coefficients = _ols_fit(design, exact_y[exact_valid])
                values[index] = (
                    coefficients[0]
                    + current_x[index] * coefficients[1]
                )

            prediction_start = group_offset + window + start
            prediction_end = group_offset + window + end
            prediction[prediction_start:prediction_end] = values
        group_offset = group_end

    if data.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}
        )
    return (
        data.select(TIME, ASSET_ID)
        .with_columns(
            pl.Series(VALUE, prediction, nan_to_null=True)
        )
        .sort([TIME, ASSET_ID])
    )


def _window_sum(
    values: np.ndarray,
    window: int,
    output_size: int,
) -> np.ndarray:
    """Return fixed-size rolling sums without materializing window views."""

    prefix = np.empty(
        (len(values) + 1, *values.shape[1:]),
        dtype=values.dtype,
    )
    prefix[0] = 0
    np.cumsum(values, axis=0, out=prefix[1:])
    return prefix[window : window + output_size] - prefix[:output_size]


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


@composer
def rolling_ols(
    target: pl.DataFrame, *factors: pl.DataFrame, window: int
) -> pl.DataFrame:
    return _rolling_ols(target, factors, window=window)


@composer
def rolling_ridge(
    target: pl.DataFrame,
    *factors: pl.DataFrame,
    window: int,
    alpha: float = 1.0,
) -> pl.DataFrame:
    _validate_non_negative_real(alpha, name="rolling_ridge alpha")
    _validate_window(window, window)
    data, factor_columns = _joined_regression_data(target, factors)
    return _rolling_gram_regression(
        data,
        factor_columns=factor_columns,
        window=window,
        method="ridge",
        alpha=float(alpha),
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
    _validate_window(window, window)
    data, factor_columns = _joined_regression_data(target, factors)
    return _rolling_gram_regression(
        data,
        factor_columns=factor_columns,
        window=window,
        method="elastic",
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        max_iter=max_iter,
        tolerance=float(tolerance),
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
