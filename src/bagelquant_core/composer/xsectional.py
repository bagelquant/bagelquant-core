"""Cross-sectional composers."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..frame import ASSET_ID, TIME, VALUE, panel_like
from .core import composer


def _grouped(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({VALUE: "x"}).join(
        group.rename({VALUE: "group"}),
        on=[TIME, ASSET_ID],
        how="inner",
    )


@composer
def orthogonalize(frame: pl.DataFrame, *factors: pl.DataFrame) -> pl.DataFrame:
    data = frame.rename({VALUE: "target"})
    for index, factor in enumerate(factors):
        data = data.join(
            factor.rename({VALUE: f"f{index}"}),
            on=[TIME, ASSET_ID],
            how="inner",
        )
    rows: list[dict[str, object]] = []
    factor_columns = [f"f{index}" for index in range(len(factors))]
    for time, group in data.partition_by(TIME, as_dict=True).items():
        clean = group.drop_nulls(["target", *factor_columns])
        if clean.height <= len(factor_columns):
            for row in group.iter_rows(named=True):
                rows.append({TIME: _key(time), ASSET_ID: row[ASSET_ID], VALUE: None})
            continue
        x = np.column_stack(
            [
                np.ones(clean.height),
                *[np.array(clean[col], dtype=float) for col in factor_columns],
            ]
        )
        y = np.array(clean["target"], dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ beta
        by_asset = dict(zip(clean[ASSET_ID], residuals, strict=True))
        for row in group.iter_rows(named=True):
            rows.append(
                {
                    TIME: _key(time),
                    ASSET_ID: row[ASSET_ID],
                    VALUE: by_asset.get(row[ASSET_ID]),
                }
            )
    return pl.DataFrame(rows).sort([TIME, ASSET_ID])


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
    return panel_like(
        data,
        pl.col("x").rank("average").over(TIME, "group")
        / pl.col("x").count().over(TIME, "group"),
    )


@composer
def group_percentile(frame: pl.DataFrame, group: pl.DataFrame) -> pl.DataFrame:
    return group_rankpct(frame, group)


def _key(value: object) -> object:
    if isinstance(value, tuple):
        return value[0]
    if isinstance(value, list):
        return value[0]
    return value
