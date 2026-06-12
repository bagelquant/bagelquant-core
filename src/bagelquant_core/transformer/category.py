"""Category-aware cross-sectional operations."""

from __future__ import annotations

import polars as pl

from ..composer.core import composer
from ..frame import ASSET_ID, TIME, VALUE, panel_like


def _joined(frame: pl.DataFrame, category: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({VALUE: "x"}).join(
        category.rename({VALUE: "category"}),
        on=[TIME, ASSET_ID],
        how="inner",
    )


@composer
def category_demean(frame: pl.DataFrame, category: pl.DataFrame) -> pl.DataFrame:
    data = _joined(frame, category)
    return panel_like(data, pl.col("x") - pl.col("x").mean().over(TIME, "category"))


@composer
def category_mean(frame: pl.DataFrame, category: pl.DataFrame) -> pl.DataFrame:
    data = _joined(frame, category)
    return panel_like(data, pl.col("x").mean().over(TIME, "category"))


@composer
def category_rank(frame: pl.DataFrame, category: pl.DataFrame) -> pl.DataFrame:
    data = _joined(frame, category)
    return panel_like(data, pl.col("x").rank("average").over(TIME, "category"))


@composer
def category_zscore(frame: pl.DataFrame, category: pl.DataFrame) -> pl.DataFrame:
    data = _joined(frame, category)
    return panel_like(
        data,
        (pl.col("x") - pl.col("x").mean().over(TIME, "category"))
        / pl.col("x").std(ddof=1).over(TIME, "category"),
    )
