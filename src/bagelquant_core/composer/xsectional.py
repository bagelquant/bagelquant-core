"""Cross-sectional composers."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..frame import (
    ASSET_ID,
    TIME,
    VALUE,
    _balanced_inner_join,
    panel_like,
)
from ..transformer.core import transformer

_ORTHOGONALIZE_CONDITION_LIMIT = 1e-3


def _grouped(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({VALUE: "x"}).join(
        group.rename({VALUE: "group"}),
        on=[TIME, ASSET_ID],
        how="inner",
    )


@transformer
def orthogonalize(
    frame: pl.DataFrame,
    *,
    factors: tuple[pl.DataFrame, ...],
    fit_intercept: bool = False,
) -> pl.DataFrame:
    if not factors:
        raise ValueError("orthogonalize requires at least one factor")
    if len(factors) == 1:
        return _orthogonalize_one_factor(
            frame,
            factors[0],
            fit_intercept=fit_intercept,
        )

    factor_columns = [f"f{index}" for index in range(len(factors))]
    data = _balanced_inner_join(
        [
            frame.rename({VALUE: "target"}),
            *[
                factor.rename({VALUE: column})
                for factor, column in zip(
                    factors,
                    factor_columns,
                    strict=True,
                )
            ],
        ]
    ).sort([TIME, ASSET_ID])
    return _orthogonalize_data(
        data,
        factor_columns,
        fit_intercept=fit_intercept,
    )


def _orthogonalize_aligned(
    frames: tuple[pl.DataFrame, ...],
    *,
    fit_intercept: bool = False,
    group_offsets: np.ndarray | None = None,
) -> pl.DataFrame:
    """Orthogonalize executor-proven positionally aligned frame values."""

    if len(frames) < 2:
        raise ValueError("orthogonalize requires at least one factor")
    if len(frames) == 2:
        return _orthogonalize_one_factor(
            frames[0],
            frames[1],
            fit_intercept=fit_intercept,
        )
    factor_columns = [
        f"f{index}" for index in range(len(frames) - 1)
    ]
    data = frames[0].select(TIME, ASSET_ID).with_columns(
        frames[0].get_column(VALUE).alias("target"),
        *[
            frame.get_column(VALUE).alias(column)
            for frame, column in zip(
                frames[1:],
                factor_columns,
                strict=True,
            )
        ],
    )
    return _orthogonalize_data(
        data,
        factor_columns,
        fit_intercept=fit_intercept,
        group_offsets=group_offsets,
    )


def _orthogonalize_data(
    data: pl.DataFrame,
    factor_columns: list[str],
    *,
    fit_intercept: bool,
    group_offsets: np.ndarray | None = None,
) -> pl.DataFrame:
    target = data.get_column("target").to_numpy().astype(float, copy=False)
    features = np.column_stack(
        [
            data.get_column(column).to_numpy().astype(float, copy=False)
            for column in factor_columns
        ]
    )
    residuals = np.full(len(data), np.nan, dtype=float)
    offsets = group_offsets
    if offsets is None:
        lengths = (
            data.group_by(TIME, maintain_order=True)
            .len()
            .get_column("len")
            .to_numpy()
        )
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    for offset, end in zip(
        offsets[:-1],
        offsets[1:],
        strict=True,
    ):
        length = end - offset
        group_y = target[offset:end]
        group_x = features[offset:end]
        valid = np.isfinite(group_y) & np.isfinite(group_x).all(axis=1)
        coefficient_count = len(factor_columns) + int(fit_intercept)
        if valid.sum() >= coefficient_count:
            design = (
                np.column_stack([np.ones(valid.sum()), group_x[valid]])
                if fit_intercept
                else group_x[valid]
            )
            gram = design.T @ design
            eigenvalues = np.maximum(
                np.linalg.eigvalsh(gram),
                0.0,
            )
            singular_values = np.sqrt(eigenvalues)
            cutoff = (
                np.finfo(np.float64).eps
                * max(len(design), design.shape[1])
                * singular_values[-1]
            )
            if (
                singular_values[0] > cutoff
                and singular_values[0]
                > _ORTHOGONALIZE_CONDITION_LIMIT
                * singular_values[-1]
            ):
                coefficients = np.linalg.solve(
                    gram,
                    design.T @ group_y[valid],
                )
            else:
                coefficients = np.linalg.lstsq(
                    design,
                    group_y[valid],
                    rcond=None,
                )[0]
            group_residuals = np.full(int(length), np.nan, dtype=float)
            group_residuals[valid] = (
                group_y[valid] - design @ coefficients
            )
            residuals[offset:end] = group_residuals
    return data.select(TIME, ASSET_ID).with_columns(
        pl.Series(VALUE, residuals, nan_to_null=True)
    )


def _orthogonalize_one_factor(
    frame: pl.DataFrame,
    factor: pl.DataFrame,
    *,
    fit_intercept: bool,
) -> pl.DataFrame:
    data = frame.rename({VALUE: "target"}).join(
        factor.rename({VALUE: "factor"}),
        on=[TIME, ASSET_ID],
        how="inner",
    )
    valid = (
        pl.col("target").is_not_null()
        & pl.col("factor").is_not_null()
        & ~pl.col("target").is_nan()
        & ~pl.col("factor").is_nan()
    )
    data = data.with_columns(
        pl.when(valid).then(pl.col("target")).otherwise(None).alias("y"),
        pl.when(valid).then(pl.col("factor")).otherwise(None).alias("x"),
    )
    n = pl.col("x").count().over(TIME)
    sum_x = pl.col("x").sum().over(TIME)
    sum_y = pl.col("y").sum().over(TIME)
    sum_xx = (pl.col("x") * pl.col("x")).sum().over(TIME)
    sum_xy = (pl.col("x") * pl.col("y")).sum().over(TIME)
    if fit_intercept:
        centered_xx = sum_xx - (sum_x * sum_x) / n
        centered_xy = sum_xy - (sum_x * sum_y) / n
        mean_y = sum_y / n
        slope = pl.when(centered_xx == 0).then(0.0).otherwise(
            centered_xy / centered_xx
        )
        intercept = mean_y - slope * (sum_x / n)
        fitted = intercept + slope * pl.col("factor")
    else:
        slope = pl.when(sum_xx == 0).then(0.0).otherwise(sum_xy / sum_xx)
        fitted = slope * pl.col("factor")
    residual = pl.when(valid & (n >= (2 if fit_intercept else 1))).then(
        pl.col("target") - fitted
    )
    return panel_like(data, residual)


@transformer
def group_rank(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").rank("average").over(TIME, "group"))


@transformer
def group_mean(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").mean().over(TIME, "group"))


@transformer
def group_max(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").max().over(TIME, "group"))


@transformer
def group_min(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").min().over(TIME, "group"))


@transformer
def group_median(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").median().over(TIME, "group"))


@transformer
def group_std(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").std(ddof=1).over(TIME, "group"))


@transformer
def group_demean(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x") - pl.col("x").mean().over(TIME, "group"))


@transformer
def group_zscore(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(
        data,
        (pl.col("x") - pl.col("x").mean().over(TIME, "group"))
        / pl.col("x").std(ddof=1).over(TIME, "group"),
    )


@transformer
def group_rankpct(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    rank = pl.col("x").rank("dense").over(TIME, "group")
    count = pl.col("x").n_unique().over(TIME, "group")
    return panel_like(data, rank / count)


@transformer
def group_percentile(frame: pl.DataFrame, *, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(
        data,
        pl.col("x").rank("average").over(TIME, "group")
        / pl.col("x").count().over(TIME, "group"),
    )
