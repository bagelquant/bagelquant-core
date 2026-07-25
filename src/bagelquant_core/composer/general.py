"""General composers."""

from __future__ import annotations

from numbers import Real

import polars as pl

from ..frame import VALUE, nary, panel_like
from .core import composer


@composer
def project(frame: pl.DataFrame, binary: pl.DataFrame) -> pl.DataFrame:
    data = frame.rename({VALUE: "x"}).join(
        binary.rename({VALUE: "binary"}),
        on=["time", "asset_id"],
        how="inner",
    )
    return panel_like(
        data,
        pl.when(pl.col("binary") == 1.0).then(pl.col("x")).otherwise(None),
    )


@composer
def mask(
    frame: pl.DataFrame,
    mask_frame: pl.DataFrame,
    *,
    replace_value: float = float("nan"),
) -> pl.DataFrame:
    if not isinstance(replace_value, Real) or isinstance(replace_value, bool):
        raise TypeError("mask replace_value must be a real number")
    data = frame.rename({VALUE: "x"}).join(
        mask_frame.rename({VALUE: "mask"}),
        on=["time", "asset_id"],
        how="inner",
    )
    condition = pl.col("mask").fill_nan(None).fill_null(0.0).cast(pl.Boolean)
    return panel_like(
        data,
        pl.when(condition)
        .then(pl.col("x"))
        .otherwise(pl.lit(float(replace_value))),
    )


@composer
def coalesce(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(frames, lambda values: pl.coalesce(values))
