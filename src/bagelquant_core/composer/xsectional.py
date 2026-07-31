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
from .core import composer


def _grouped(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({VALUE: "x"}).join(
        group.rename({VALUE: "group"}),
        on=[TIME, ASSET_ID],
        how="inner",
    )


@composer
def orthogonalize(frame: pl.DataFrame, *factors: pl.DataFrame) -> pl.DataFrame:
    if not factors:
        raise ValueError("orthogonalize requires at least one factor")
    if len(factors) == 1:
        return _orthogonalize_one_factor(frame, factors[0])

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
    target = data.get_column("target").to_numpy().astype(float, copy=False)
    features = np.column_stack(
        [
            data.get_column(column).to_numpy().astype(float, copy=False)
            for column in factor_columns
        ]
    )
    residuals = np.full(len(data), np.nan, dtype=float)
    lengths = (
        data.group_by(TIME, maintain_order=True)
        .len()
        .get_column("len")
        .to_numpy()
    )
    offset = 0
    for length in lengths:
        end = offset + int(length)
        group_y = target[offset:end]
        group_x = features[offset:end]
        valid = np.isfinite(group_y) & np.isfinite(group_x).all(axis=1)
        if valid.sum() > len(factor_columns):
            design = np.column_stack(
                [
                    np.ones(valid.sum()),
                    group_x[valid],
                ]
            )
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
        offset = end
    return data.select(TIME, ASSET_ID).with_columns(
        pl.Series(VALUE, residuals, nan_to_null=True)
    )


def _orthogonalize_one_factor(frame: pl.DataFrame, factor: pl.DataFrame) -> pl.DataFrame:
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
    centered_xx = sum_xx - (sum_x * sum_x) / n
    centered_xy = sum_xy - (sum_x * sum_y) / n
    mean_y = sum_y / n
    slope = pl.when(centered_xx == 0).then(0.0).otherwise(centered_xy / centered_xx)
    intercept = mean_y - slope * (sum_x / n)
    residual = pl.when(valid & (n > 1)).then(
        pl.col("target") - (intercept + slope * pl.col("factor"))
    )
    return panel_like(data, residual)


@composer
def group_rank(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").rank("average").over(TIME, "group"))


@composer
def group_mean(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").mean().over(TIME, "group"))


@composer
def group_max(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").max().over(TIME, "group"))


@composer
def group_min(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").min().over(TIME, "group"))


@composer
def group_median(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").median().over(TIME, "group"))


@composer
def group_std(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x").std(ddof=1).over(TIME, "group"))


@composer
def group_demean(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(data, pl.col("x") - pl.col("x").mean().over(TIME, "group"))


@composer
def group_zscore(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(
        data,
        (pl.col("x") - pl.col("x").mean().over(TIME, "group"))
        / pl.col("x").std(ddof=1).over(TIME, "group"),
    )


@composer
def group_rankpct(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    rank = pl.col("x").rank("dense").over(TIME, "group")
    count = pl.col("x").n_unique().over(TIME, "group")
    return panel_like(data, rank / count)


@composer
def group_percentile(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    data = _grouped(frame, group)
    return panel_like(
        data,
        pl.col("x").rank("average").over(TIME, "group")
        / pl.col("x").count().over(TIME, "group"),
    )
