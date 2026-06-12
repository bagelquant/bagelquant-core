"""General composers."""

from __future__ import annotations

import polars as pl

from ..frame import VALUE, nary, panel_like
from .core import composer


@composer
def project(frame: pl.DataFrame, binary: pl.DataFrame) -> pl.DataFrame:
    return mask(frame, binary)


@composer
def mask(
    frame: pl.DataFrame, binary: pl.DataFrame, *, keep_value: float = 1.0
) -> pl.DataFrame:
    data = frame.rename({VALUE: "x"}).join(
        binary.rename({VALUE: "mask"}),
        on=["time", "asset_id"],
        how="inner",
    )
    return panel_like(
        data, pl.when(pl.col("mask") == keep_value).then(pl.col("x")).otherwise(None)
    )


@composer
def coalesce(*frames: pl.DataFrame) -> pl.DataFrame:
    return nary(frames, lambda values: pl.coalesce(values))
