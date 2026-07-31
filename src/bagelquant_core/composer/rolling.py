"""Rolling multi-input composers."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from ..frame import ASSET_ID, TIME, VALUE, panel_like
from .core import composer

_MAX_WORKING_BYTES = 64 * 1024 * 1024
_OLS_ARRAY_COPIES = 4
_SINGLE_FACTOR_OLS_ARRAYS = 16


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

    if len(factor_columns) == 1:
        return _rolling_single_factor_ols(
            data,
            factor_column=factor_columns[0],
            window=window,
        )

    output: list[pl.DataFrame] = []
    feature_count = len(factor_columns)
    coefficient_count = feature_count + 1
    bytes_per_prediction = (
        window
        * coefficient_count
        * np.dtype(np.float64).itemsize
        * _OLS_ARRAY_COPIES
    )
    rows_per_batch = max(
        1,
        _MAX_WORKING_BYTES // max(bytes_per_prediction, 1),
    )

    for group in data.partition_by(ASSET_ID):
        target_values = group.get_column("target").to_numpy().astype(
            float, copy=False
        )
        features = np.column_stack(
            [
                group.get_column(column).to_numpy().astype(float, copy=False)
                for column in factor_columns
            ]
        )
        prediction = np.full(len(group), np.nan, dtype=float)
        prediction_count = len(group) - window
        if prediction_count > 0:
            target_windows = sliding_window_view(target_values, window)
            feature_windows = sliding_window_view(
                features,
                (window, feature_count),
            )[:, 0, :, :]

            for start in range(0, prediction_count, rows_per_batch):
                end = min(start + rows_per_batch, prediction_count)
                train_y = target_windows[start:end]
                train_x = feature_windows[start:end]
                current_x = features[window + start : window + end]
                valid = np.isfinite(train_y) & np.isfinite(train_x).all(axis=2)

                design = np.zeros(
                    (end - start, window, coefficient_count),
                    dtype=float,
                )
                design[:, :, 0] = valid
                design[:, :, 1:] = np.where(valid[:, :, None], train_x, 0.0)
                valid_target = np.where(valid, train_y, 0.0)
                valid_counts = valid.sum(axis=1)
                coefficients, ambiguous = _batched_lstsq(
                    design,
                    valid_target,
                    row_counts=valid_counts,
                )
                eligible = valid.any(axis=1) & np.isfinite(current_x).all(axis=1)

                for index in np.flatnonzero(ambiguous & eligible):
                    selected = valid[index]
                    exact_design = np.column_stack(
                        [
                            np.ones(selected.sum()),
                            train_x[index][selected],
                        ]
                    )
                    coefficients[index] = _ols_fit(
                        exact_design,
                        train_y[index][selected],
                    )

                current_design = np.column_stack(
                    [np.ones(end - start), current_x]
                )
                values = np.einsum(
                    "ij,ij->i",
                    current_design,
                    coefficients,
                )
                prediction[window + start : window + end] = np.where(
                    eligible,
                    values,
                    np.nan,
                )

        output.append(
            group.select(TIME, ASSET_ID).with_columns(
                pl.Series(VALUE, prediction, nan_to_null=True)
            )
        )

    if not output:
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}
        )
    return pl.concat(output).sort([TIME, ASSET_ID])


def _rolling_single_factor_ols(
    data: pl.DataFrame,
    *,
    factor_column: str,
    window: int,
) -> pl.DataFrame:
    """Predict single-factor OLS from prior-window sufficient statistics."""

    output: list[pl.DataFrame] = []
    itemsize = np.dtype(np.float64).itemsize
    rows_per_batch = max(
        1,
        _MAX_WORKING_BYTES // (itemsize * _SINGLE_FACTOR_OLS_ARRAYS) - window,
    )
    epsilon = np.finfo(np.float64).eps

    for group in data.partition_by(ASSET_ID):
        target_values = group.get_column("target").to_numpy().astype(
            float, copy=False
        )
        factor_values = group.get_column(factor_column).to_numpy().astype(
            float, copy=False
        )
        prediction = np.full(len(group), np.nan, dtype=float)
        prediction_count = len(group) - window

        for start in range(0, max(prediction_count, 0), rows_per_batch):
            end = min(start + rows_per_batch, prediction_count)
            batch_size = end - start
            train_y = target_values[start : window + end]
            train_x = factor_values[start : window + end]
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
            current_x = factor_values[window + start : window + end]
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
                exact_y = target_values[window_start : window_start + window]
                exact_x = factor_values[window_start : window_start + window]
                exact_valid = np.isfinite(exact_y) & np.isfinite(exact_x)
                design = np.column_stack(
                    [np.ones(exact_valid.sum()), exact_x[exact_valid]]
                )
                coefficients = _ols_fit(design, exact_y[exact_valid])
                values[index] = (
                    coefficients[0]
                    + current_x[index] * coefficients[1]
                )

            prediction[window + start : window + end] = values

        output.append(
            group.select(TIME, ASSET_ID).with_columns(
                pl.Series(VALUE, prediction, nan_to_null=True)
            )
        )

    if not output:
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, VALUE: pl.Float64}
        )
    return pl.concat(output).sort([TIME, ASSET_ID])


def _window_sum(
    values: np.ndarray,
    window: int,
    output_size: int,
) -> np.ndarray:
    """Return fixed-size rolling sums without materializing window views."""

    prefix = np.empty(len(values) + 1, dtype=values.dtype)
    prefix[0] = 0
    np.cumsum(values, out=prefix[1:])
    return prefix[window : window + output_size] - prefix[:output_size]


def _batched_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    *,
    row_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve minimum-norm least squares with NumPy's default rank cutoff."""

    left, singular_values, right = np.linalg.svd(
        design,
        full_matrices=False,
    )
    relative_cutoff = np.finfo(design.dtype).eps * np.maximum(
        row_counts,
        design.shape[-1],
    )
    cutoff = relative_cutoff[:, None] * singular_values[:, :1]
    inverse = np.divide(
        1.0,
        singular_values,
        out=np.zeros_like(singular_values),
        where=singular_values > cutoff,
    )
    projected = np.einsum("bwi,bw->bi", left, target)
    coefficients = np.einsum(
        "bij,bj->bi",
        right.swapaxes(-2, -1),
        inverse * projected,
    )
    tolerance = np.maximum(cutoff, np.finfo(design.dtype).tiny) * 1e-8
    ambiguous = (np.abs(singular_values - cutoff) <= tolerance).any(axis=1)
    return coefficients, ambiguous


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
    return _rolling_ols(target, factors, window=window)


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
